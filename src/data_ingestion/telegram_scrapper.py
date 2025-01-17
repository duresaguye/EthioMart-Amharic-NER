import os
import json
from telethon import TelegramClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_id = os.getenv('API_ID')
api_hash = os.getenv('API_HASH')
phone = os.getenv('PHONE')

client = TelegramClient('session_name', api_id, api_hash)

async def main():
    await client.start(phone)
    channel_names = ['@modernshoppingcenter', '@Shewabrand', '@helloomarketethiopia','@sinayelj','ZemenEskender']
    os.makedirs('data/raw', exist_ok=True)
    
    for channel_name in channel_names:
        channel = await client.get_entity(channel_name)
        messages = await client.get_messages(channel, limit=100)
        data = []
        for message in messages:
            data.append({
                'id': message.id,
                'date': message.date.isoformat(),  # Convert datetime to string
                'message': message.message,
                'sender_id': message.sender_id
            })
        with open(f'data/raw/{channel_name[1:]}.json', 'w') as f:  # Remove '@' from filename
            json.dump(data, f, ensure_ascii=False, indent=4)

with client:
    client.loop.run_until_complete(main())