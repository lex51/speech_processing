import os
from celery import Celery

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, ContentType, File
from aiogram.utils.markdown import hbold
from loguru import logger as lg
from pathlib import Path

# Bot token can be obtained via https://t.me/BotFather
TOKEN = os.environ.get("BOT_TOKEN")

passcode = os.environ.get("REDIS_PASS")
CELERY_BROKER_URL = (
    os.environ.get("CELERY_BROKER_URL", "redis://:" + passcode + "@localhost:6379"),
)
CELERY_RESULT_BACKEND = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://:" + passcode + "@localhost:6379"
)


celery = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
# All handlers should be attached to the Router (or Dispatcher)
dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    # Most event objects have aliases for API methods that can be called in events' context
    # For example if you want to answer to incoming message you can use `message.answer(...)` alias
    # and the target chat will be passed to :ref:`aiogram.methods.send_message.SendMessage`
    # method automatically or call API method directly via
    # Bot instance: `bot.send_message(chat_id=message.chat.id, ...)`
    await message.answer(f"Hello, {hbold(message.from_user.full_name)}!")


async def handle_file(file: File, file_name: str, path: str):
    Path(f"{path}").mkdir(parents=True, exist_ok=True)

    file = await dp.download_file(
        file_path=file.file_path, destination=f"{path}/{file_name}"
    )
    return file


@dp.message_handler(content_types=[ContentType.VOICE])
async def echo_handler(message: types.Message) -> None:
    voice = await message.voice.get_file()
    path = "/files/voices"

    file = await handle_file(file=voice, file_name=f"{voice.file_id}.ogg", path=path)
    processed_file = await celery.send_task("tasks.get_emoji", args=[file])
    lg.info(f"task {processed_file.id}")
    # returns dict
    return processed_file


async def main() -> None:
    # Initialize Bot instance with a default parse mode which will be passed to all API calls
    bot = Bot(TOKEN, parse_mode=ParseMode.HTML)
    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
