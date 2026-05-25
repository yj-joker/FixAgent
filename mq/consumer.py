"""
RabbitMQ 消费者

监听三组队列：
- memory.realtime.queue    → 实时记忆更新（prefetch=5）
- memory.consolidate.queue → 记忆整合（prefetch=1，单任务耗时长）
- knowledge.import.queue   → 知识库导入（prefetch=1，PDF解析+向量化耗时长）

处理完后将结果发到对应 result 队列，由 Java 端消费并更新状态。
"""

import json
import logging
import asyncio

import aio_pika
import httpx

from mq.connection import get_connection
from config.settings import get_settings

logger = logging.getLogger(__name__)

# ===== 记忆系统队列 =====
EXCHANGE_NAME = "memory.exchange"
RESULT_KEY = "memory.result"
REALTIME_QUEUE = "memory.realtime.queue"
CONSOLIDATE_QUEUE = "memory.consolidate.queue"
RESULT_QUEUE = "memory.result.queue"

# ===== 知识导入队列 =====
KNOWLEDGE_EXCHANGE = "knowledge.exchange"
KNOWLEDGE_IMPORT_QUEUE = "knowledge.import.queue"
KNOWLEDGE_RESULT_KEY = "knowledge.result"
KNOWLEDGE_RESULT_QUEUE = "knowledge.result.queue"


async def publish_result(channel: aio_pika.abc.AbstractChannel, data: dict,
                         exchange_name: str = EXCHANGE_NAME, routing_key: str = RESULT_KEY):
    exchange = await channel.get_exchange(exchange_name)
    await exchange.publish(
        aio_pika.Message(
            body=json.dumps(data, ensure_ascii=False).encode(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=routing_key,
    )


async def handle_realtime(message: aio_pika.abc.AbstractIncomingMessage, channel: aio_pika.abc.AbstractChannel):
    async with message.process(requeue=False):
        body = json.loads(message.body)
        session_id = str(body["sessionId"])
        user_id = body["userId"]
        logger.info("[MQ消费] 实时更新开始, 会话ID:%s", session_id)

        try:
            from agents.realtime_memory_agent import get_realtime_memory_agent
            from agents.base_agent import AgentInput

            agent = get_realtime_memory_agent()
            input_data = AgentInput(
                user_message=body["userMessage"],
                session_id=session_id,
                context={
                    "user_message": body["userMessage"],
                    "ai_response": body.get("aiResponse", ""),
                    "recent_facts": [],
                },
            )
            result = await agent.run(input_data)
            result_data = result.metadata.get("result", {})

            await publish_result(channel, {
                "type": "realtime_update",
                "sessionId": session_id,
                "userId": user_id,
                "currentRound": body.get("currentRound"),
                "success": True,
                "data": result_data,
            })
            logger.info("[MQ消费] 实时更新完成, 会话ID:%s, has_update=%s",
                        session_id, result_data.get("has_update", False))

        except Exception as e:
            logger.error("[MQ消费] 实时更新失败, 会话ID:%s, 错误:%s", session_id, e)
            await publish_result(channel, {
                "type": "realtime_update",
                "sessionId": session_id,
                "userId": user_id,
                "success": False,
                "error": str(e),
                "data": {},
            })


async def handle_consolidate(message: aio_pika.abc.AbstractIncomingMessage, channel: aio_pika.abc.AbstractChannel):
    async with message.process(requeue=False):
        body = json.loads(message.body)
        session_id = str(body["sessionId"])
        user_id = body["userId"]
        round_count = body["roundCount"]
        max_memory = body["maxMemory"]
        logger.info("[MQ消费] 记忆整合开始, 会话ID:%s, 轮次:%s", session_id, round_count)

        try:
            settings = get_settings()

            # 从 Java 端拉取整合参数（携带内部鉴权令牌）
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{settings.java_service_url}/weixiu/memory/consolidation-params",
                    params={
                        "sessionId": session_id,
                        "userId": user_id,
                        "roundCount": round_count,
                        "maxMemory": max_memory,
                    },
                    headers={"X-Internal-Token": settings.internal_token},
                )
                resp.raise_for_status()
                api_result = resp.json()

            params = api_result.get("data")
            if not params:
                logger.info("[MQ消费] 无需整合（Java返回空）, 会话ID:%s", session_id)
                return

            # 调用已有的 memory_agent
            from agents.memory_agent import get_memory_agent
            from agents.base_agent import AgentInput

            conv_dicts = []
            for i, m in enumerate(params.get("memoryMessages", [])):
                conv_dicts.append({"seq": i + 1, "role": m["role"], "content": m["content"]})

            agent_input = AgentInput(
                user_message="请整理以下对话记录",
                session_id=session_id,
                context={
                    "conversations": conv_dicts,
                    "old_preferences": params.get("memoryPreferenceVOList", []),
                    "old_unresolved": params.get("memoryUnresolvedVOList", []),
                    "previous_summary": params.get("previousSummary"),
                },
            )

            result = await get_memory_agent().run(agent_input)

            if result.metadata.get("status") == "error":
                raise RuntimeError(result.metadata.get("error_detail", "整合失败"))

            summary_data = result.metadata.get("summary", {})
            summary_data["consolidatedMessageIds"] = params.get("messageIds", [])

            await publish_result(channel, {
                "type": "consolidation",
                "sessionId": session_id,
                "userId": user_id,
                "success": True,
                "data": summary_data,
            })
            logger.info("[MQ消费] 记忆整合完成, 会话ID:%s", session_id)

        except Exception as e:
            logger.error("[MQ消费] 记忆整合失败, 会话ID:%s, 错误:%s", session_id, e, exc_info=True)
            await publish_result(channel, {
                "type": "consolidation",
                "sessionId": session_id,
                "userId": user_id,
                "success": False,
                "error": str(e),
                "data": {},
            })


async def handle_knowledge_import(message: aio_pika.abc.AbstractIncomingMessage, channel: aio_pika.abc.AbstractChannel):
    """消费知识导入任务：解析文档 → 向量化 → 存入 Redis 向量库"""
    async with message.process(requeue=False):
        body = json.loads(message.body)
        task_id = body.get("taskId", "unknown")
        file_url = body.get("fileUrl", "")
        file_type = body.get("fileType", "pdf")
        category = body.get("category")
        user_id = body.get("userId")
        logger.info("[MQ消费] 知识导入开始, taskId=%s, fileUrl=%s", task_id, file_url)

        try:
            from tools.knowledge_retrieval_tool import get_knowledge_retrieval_tool
            from services.vector_service import get_vector_service

            # 调用已有的知识导入逻辑
            tool = get_knowledge_retrieval_tool()
            result = await tool.import_document(
                file_url=file_url,
                file_type=file_type,
                category=category,
            )

            await publish_result(channel, {
                "taskId": task_id,
                "userId": user_id,
                "success": True,
                "data": {
                    "total_chunks": result.get("total_chunks", 0),
                    "file_url": file_url,
                },
            }, exchange_name=KNOWLEDGE_EXCHANGE, routing_key=KNOWLEDGE_RESULT_KEY)

            logger.info("[MQ消费] 知识导入完成, taskId=%s, chunks=%s", task_id, result.get("total_chunks", 0))

        except Exception as e:
            logger.error("[MQ消费] 知识导入失败, taskId=%s, 错误:%s", task_id, e, exc_info=True)
            await publish_result(channel, {
                "taskId": task_id,
                "userId": user_id,
                "success": False,
                "error": str(e),
                "data": {},
            }, exchange_name=KNOWLEDGE_EXCHANGE, routing_key=KNOWLEDGE_RESULT_KEY)


async def _declare_topology(channel: aio_pika.abc.AbstractChannel):
    """
    声明 Exchange / Queue / Binding，与 Java 端 RabbitMQConfig 保持一致。
    declare 是幂等的：如果已存在且参数相同则直接返回，不会重复创建。
    这样 Python 和 Java 无论谁先启动都能正常工作。
    """
    # 死信
    dlx = await channel.declare_exchange(
        "memory.dlx", aio_pika.ExchangeType.FANOUT, durable=True
    )
    dlx_queue = await channel.declare_queue("memory.dlx.queue", durable=True)
    await dlx_queue.bind(dlx)

    # ===== 记忆系统拓扑 =====
    exchange = await channel.declare_exchange(
        EXCHANGE_NAME, aio_pika.ExchangeType.TOPIC, durable=True
    )

    # 实时更新队列（TTL 5min）
    realtime_q = await channel.declare_queue(
        REALTIME_QUEUE, durable=True,
        arguments={"x-message-ttl": 300_000, "x-dead-letter-exchange": "memory.dlx"},
    )
    await realtime_q.bind(exchange, "memory.realtime")

    # 整合队列（TTL 10min）
    consolidate_q = await channel.declare_queue(
        CONSOLIDATE_QUEUE, durable=True,
        arguments={"x-message-ttl": 600_000, "x-dead-letter-exchange": "memory.dlx"},
    )
    await consolidate_q.bind(exchange, "memory.consolidate")

    # 结果队列
    result_q = await channel.declare_queue(RESULT_QUEUE, durable=True)
    await result_q.bind(exchange, "memory.result")

    # ===== 知识导入拓扑 =====
    knowledge_exchange = await channel.declare_exchange(
        KNOWLEDGE_EXCHANGE, aio_pika.ExchangeType.TOPIC, durable=True
    )

    # 知识导入队列（TTL 30min，PDF解析+向量化耗时长）
    knowledge_import_q = await channel.declare_queue(
        KNOWLEDGE_IMPORT_QUEUE, durable=True,
        arguments={"x-message-ttl": 1_800_000, "x-dead-letter-exchange": "memory.dlx"},
    )
    await knowledge_import_q.bind(knowledge_exchange, "knowledge.import")

    # 知识导入结果队列
    knowledge_result_q = await channel.declare_queue(
        KNOWLEDGE_RESULT_QUEUE, durable=True,
    )
    await knowledge_result_q.bind(knowledge_exchange, "knowledge.result")

    return realtime_q, consolidate_q, knowledge_import_q


async def start_consumers():
    connection = await get_connection()

    # 先用一个临时通道声明拓扑
    init_channel = await connection.channel()
    realtime_q, consolidate_q, knowledge_import_q = await _declare_topology(init_channel)
    await init_channel.close()

    # 实时更新通道（prefetch=5，允许并发处理多条）
    realtime_channel = await connection.channel()
    await realtime_channel.set_qos(prefetch_count=5)
    realtime_queue = await realtime_channel.get_queue(REALTIME_QUEUE)
    await realtime_queue.consume(
        lambda msg: handle_realtime(msg, realtime_channel)
    )

    # 记忆整合通道（prefetch=1，单任务耗时长，串行处理）
    consolidate_channel = await connection.channel()
    await consolidate_channel.set_qos(prefetch_count=1)
    consolidate_queue = await consolidate_channel.get_queue(CONSOLIDATE_QUEUE)
    await consolidate_queue.consume(
        lambda msg: handle_consolidate(msg, consolidate_channel)
    )

    # 知识导入通道（prefetch=1，PDF解析+向量化耗时长，串行处理）
    knowledge_channel = await connection.channel()
    await knowledge_channel.set_qos(prefetch_count=1)
    knowledge_queue = await knowledge_channel.get_queue(KNOWLEDGE_IMPORT_QUEUE)
    await knowledge_queue.consume(
        lambda msg: handle_knowledge_import(msg, knowledge_channel)
    )

    logger.info("[MQ消费] 消费者启动完成，监听 %s, %s, %s",
                REALTIME_QUEUE, CONSOLIDATE_QUEUE, KNOWLEDGE_IMPORT_QUEUE)
