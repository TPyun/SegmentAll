import asyncio
import telegram

class TeleSender:
    def __init__(self):
        self.token = '7185803418:AAGvbtK4oXJ1LcwaJGZMZOy3DpCsqdKW680'
        self.chat_id = 7128399973
    
    def get_bot(self):
        self.bot = telegram.Bot(token = self.token)
        
    async def send_text_async(self, text): #실행시킬 함수명 임의지정
        await self.bot.send_message(self.chat_id, text)
        
    def send_text(self, text):
        try:
            self.get_bot()
            asyncio.run(self.send_text_async(text))
        except:
            print("Telegram message sending failed")
        
    async def send_photo_async(self, photo_path):
        await self.bot.send_photo(self.chat_id, photo=open(photo_path, 'rb'))
        
    def send_photo(self, photo_path):
        try:
            self.get_bot()
            asyncio.run(self.send_photo_async(photo_path))
        except:
            print("Telegram photo sending failed")