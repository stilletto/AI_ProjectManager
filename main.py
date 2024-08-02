import time
import discord
import asyncio
import openai
import json
import wave
import os
from datetime import datetime
from discord.ext import commands
from collections import defaultdict
from openai import OpenAI
import tiktoken
import random
import traceback
import whisper

# Load configuration from config.json
with open('config.json', 'r') as f:
    config = json.load(f)

bot_token = config["bot_token"]
api_key = config["api_key"]
general_channel_name = config["general_channel_name"]
voice_channel_name = config["voice_channel_name"]
use_local_whisper = config["use_local_whisper"]
language = config["language"]

if use_local_whisper:
    whisper_model = whisper.load_model("large")

openai.api_key = api_key
client_gpt = OpenAI(api_key=api_key)

intents = discord.Intents.default()
intents.voice_states = True
intents.guilds = True
intents.members = True
bot = commands.Bot(command_prefix='!', intents=intents)

def create_json_file(audio_data):
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    json_filename = f'record_{timestamp}.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(audio_data, f, ensure_ascii=False, indent=4)
    return json_filename

class MySink(discord.sinks.WaveSink):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_data = defaultdict(bytes)

    def write(self, data, user_id):
        self.audio_data[user_id] += data

    def finish(self):
        return self.audio_data

    def cleanup(self):
        print("Custom cleanup called")

def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokens = encoding.encode(text)
    return len(tokens)

async def process_and_summarize_json_files():
    json_files_content = []

    for file_name in os.listdir():
        if file_name.endswith('.json') and file_name != 'summary.json':
            with open(file_name, 'r', encoding='utf-8') as file:
                file_content = json.load(file)
                json_files_content.append(file_content)
            print(f"File {file_name} read successfully")

    combined_content = {"data": [item for content_list in json_files_content for item in content_list if "audio_filename" not in item]}
    combined_text = json.dumps(combined_content, ensure_ascii=False, indent=0)
    print(f"Combined text: {combined_text}")

    token_count = count_tokens(combined_text)

    if token_count > 4000:
        response = openai.Completion.create(
            model="gpt-4o",
            prompt="This data is transferred for maximum compression. You need to compress them, leaving only the most important plus the names, and other unique data that can help for further participation in the projects of this company. Remove all unnecessary signs that are meaningless and related to the markup, as well as system messages, time, etc., leave only the important text. You need to respond only with this data, in the most compressed format, do not make any other comments, this data will be used through the API. Data: " + combined_text,
            max_tokens=3000
        )
        combined_text = response.choices[0].text.strip()
    elif token_count < 20:
        print("Not enough data to summarize")
        return None

    system_prompt = "It is history of conversation data. Use it to correct transcribe errors and for more precise answer. Do not use this information as a current message, this is old data and is only needed as a hint. History:"
    memory_data = {"role": "system", "content": system_prompt + combined_text}

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file_name = f'summary_{timestamp}.json'
    with open(summary_file_name, 'w', encoding='utf-8') as summary_file:
        json.dump(memory_data, summary_file, ensure_ascii=False, indent=4)

    for file_name in os.listdir():
        if file_name.endswith('.json') and file_name != summary_file_name:
            os.remove(file_name)
    print(f'Summary created and saved in {summary_file_name}')
    return memory_data

class MyClient(discord.Client):
    def __init__(self, intents, *args, **kwargs):
        super().__init__(intents=intents, *args, **kwargs)
        self.recording_users = {}
        self.sink = MySink()
        self.voice_client = None
        self.recordings_metadata = []
        self.files_saved_event = asyncio.Event()
        self.pending_tasks = []

    async def on_ready(self):
        print(f'Logged in as {self.user}!')
        guild = discord.utils.get(self.guilds)
        self.voice_channel = discord.utils.get(guild.voice_channels, name=voice_channel_name)
        if self.voice_channel:
            await self.join_voice_channel(self.voice_channel)
            self.setup_users_in_channel(self.voice_channel)
            self.loop.create_task(self.monitor_voice_activity())
        else:
            print("Voice channel not found")

    async def join_voice_channel(self, voice_channel):
        self.voice_client = await voice_channel.connect()
        print(f"Connected to {voice_channel.name}")
        if voice_channel.members:
            self.start_recording_for_all_members(voice_channel)

    def setup_users_in_channel(self, voice_channel):
        for member in voice_channel.members:
            if not member.bot:
                self.recording_users[member.id] = [0, 'recording', 0]
                print(f'Setup user {member.name} in voice channel with started recording.')

    def start_recording_for_all_members(self, voice_channel):
        for member in voice_channel.members:
            if not member.bot:
                self.recording_users[member.id] = [0, 'stopped', None]
                try:
                    self.voice_client.start_recording(self.sink, self.finished_callback, member.id)
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error starting recording 1: {e}")
                    try:
                        self.voice_client.stop_recording()
                    except Exception:
                        pass
                    try:
                        self.voice_client = voice_channel.connect()
                        self.voice_client.start_recording(self.sink, self.finished_callback, member.id)
                        print(f"Reconnected to voice channel")
                    except Exception as e:
                        traceback.print_exc()
                        print(f"Error starting recording 2: {e}")

    async def monitor_voice_activity(self):
        while True:
            for user_id in list(self.recording_users.keys()):
                current_len = len(self.sink.audio_data[user_id])
                last_len = self.recording_users[user_id][0]
                try:
                    if current_len > last_len:
                        self.recording_users[user_id][1] = 'recording'
                        self.recording_users[user_id][2] = datetime.now()
                        self.recording_users[user_id][0] = current_len
                    else:
                        if self.recording_users[user_id][1] == 'recording':
                            if (datetime.now() - self.recording_users[user_id][2]).seconds > 3:
                                task = asyncio.create_task(self.stop_recording_silence(user_id))
                                self.pending_tasks.append(task)
                                self.recording_users[user_id][1] = 'stopped'
                                self.recording_users[user_id][2] = None
                except Exception as e:
                    print(f"Error monitoring voice activity: {e}")
                    self.recording_users[user_id][1] = 'stopped'
                    self.recording_users[user_id][2] = None
                    self.recording_users[user_id][0] = 0
            await asyncio.sleep(0.05)

    async def stop_recording_silence(self, user_id):
        print(f'{user_id} stopped speaking for 3 seconds.')
        try:
            await self.finished_callback(self.sink, user_id)
        except Exception as e:
            print(f"Error stopping recording: {e}")
        self.sink.audio_data[user_id] = b''
        self.recording_users[user_id][0] = 0
        self.recording_users[user_id][1] = 'stopped'
        self.recording_users[user_id][2] = None

    async def on_voice_state_update(self, member, before, after):
        if member.bot:
            return
        if after.channel == self.voice_channel and not after.self_mute:
            print(f'{member.name} started speaking.')
            self.recording_users[member.id] = [0, 'stopped', None]
            self.sink = MySink()
            try:
                self.voice_client.start_recording(self.sink, self.finished_callback, member.id)
            except Exception as e:
                traceback.print_exc()
                print(f"Error starting recording 11: {e}")
                try:
                    self.voice_client.stop_recording()
                except Exception:
                    pass
                try:
                    try:
                        await self.voice_client.disconnect()
                    except Exception:
                        pass
                    try:
                        self.voice_client = await self.voice_channel.connect()
                    except Exception:
                        pass
                    self.voice_client.start_recording(self.sink, self.finished_callback, member.id)
                    print(f"Reconnected to voice channel")
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error starting recording 12: {e}")
        elif before.channel == self.voice_channel and (after.self_mute or after.channel != self.voice_channel):
            if member.id not in self.recording_users:
                self.recording_users[member.id] = [0, 'stopped', None]
            else:
                self.recording_users[member.id][1] = 'stopped'
            print(f'{member.name} stopped speaking or left the channel.')
            task = asyncio.create_task(self.stop_recording_silence(member.id))
            self.pending_tasks.append(task)
            remaining_members = [m for m in self.voice_channel.members if not m.bot]
            if len(remaining_members) < 1:
                print("No members left in voice channel, meeting is over.")
                for user_id in list(self.recording_users.keys()):
                    if self.recording_users[user_id][1] == 'recording':
                        await self.finished_callback(self.sink, user_id)
                    try:
                        await self.stop_recording_silence(user_id)
                    except Exception as e:
                        print(f"Error stopping recording: {e}")
                    self.sink.audio_data[user_id] = b''
                    self.recording_users[user_id][0] = 0
                    self.recording_users[user_id][1] = 'stopped'
                    self.recording_users[user_id][2] = None

                tasks = [task for task in self.pending_tasks if not task.done()]

                if tasks:
                    await asyncio.gather(*tasks)

                self.files_saved_event.set()

                print("All recordings stopped, summarizing meeting...")
                await self.summarize_meeting_and_send()

    async def finished_callback(self, sink, user_id):
        print("Finished recording, processing data...")
        audio_data = sink.audio_data.pop(user_id, None)
        sink.cleanup()
        if audio_data:
            member = self.voice_channel.guild.get_member(user_id)
            task = asyncio.create_task(self.save_recording(member, audio_data))
            self.pending_tasks.append(task)
        else:
            print(f'No audio data was saved for {user_id}')

    async def save_recording(self, member, audio_data):
        if not audio_data:
            print(f"No audio data for {member.name}")
            return
        print(f"Audio data length for {member.name}: {len(audio_data)}")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        directory = f"recordings/{member.name}"
        os.makedirs(directory, exist_ok=True)
        wav_filename = os.path.join(directory, f'audio_{timestamp}.wav')
        print(f"Saving audio data for {member.name} to {wav_filename}")
        with wave.open(wav_filename, 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(audio_data)
        print(f'Audio recorded to {wav_filename}')
        audio_metadata = {
            'timestamp': timestamp,
            'audio_filename': wav_filename,
            'username': member.name
        }
        self.recordings_metadata.append(audio_metadata)
        self.pending_tasks.remove(asyncio.current_task())

    def transcribe_audio(self, audio_file_path, chat_prompt):
        if use_local_whisper:
            return self.transcribe_audio_local(audio_file_path, chat_prompt)
        else:
            return self.transcribe_audio_online(audio_file_path, chat_prompt)


    def transcribe_audio_local(self, audio_file_path, chat_prompt):
        audio = whisper.load_audio(audio_file_path)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(whisper_model.device)

        # detect the spoken language
        #_, probs = whisper_model.detect_language(mel)
        #lg = max(probs, key=probs.get)
        lg = "ru"
        print(f"Detected language: {lg}")

        # decode the audio
        options = whisper.DecodingOptions(language=lg, prompt=chat_prompt)
        result = whisper.decode(whisper_model, mel, options)

        # print the recognized textu
        print(result.text)
        return str(result.text)

    def transcribe_audio_online(self, audio_file_path, chat_prompt):
        with open(audio_file_path, 'rb') as audio_file:
            response = client_gpt.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format='text',
                language=language,
                prompt=chat_prompt
            )
        print(response)
        return str(response)

    def work_around_gpt(self, transcribed, history=""):
        messages = [
            {"role": "system", "content": f"You play the role of an assistant in text recognition, in this case there are 5 options for recognizing the same phrase, you must understand which one is the most correct, in addition, correct errors if you see that the meaning of the recognition is not what it should be. To help you, the call history in the form of text for this group of people is given. The following is the story:{history}"},
            {"role": "system", "content": "Please note that in your response you must strictly return only one message from the list of recognized ones from the user, without any additional formatting, do not add your comments, why you chose it and how you corrected it. Only the message, as Whisper should have recognized it if it had recognized everything correctly."},
            {"role": "user", "content": str(transcribed)},
        ]
        response = client_gpt.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            messages=messages
        )
        return response.choices[0].message.content

    async def fix_chat_prompt(self, chat_prompt, history=""):
        if not chat_prompt or len(chat_prompt) < 6:
            return ""
        fix_prompt = "The following is a official meeting transcript. I need it prepare to be context in future speach recognition. Fix errors in this chat and return it in Open AI Whisper prompt format without any comments. Compress output to 225 tokens. And it must be in uderstandable for Whister STT model. It will be used in Whisper model as prompt. Pass names in the prompt to prevent misspellings. Example of prompt: Glossary: Aimee,Shawn,BBQ,Whisky,Doughnuts menu,Omelet,Gilbert Moan,Asics"

        msg = [{"role": "system", "content": fix_prompt}]
        if len(history) > 6:
               msg.append({"role": "system", "content": f"Old messages history to help you create best prompt: {history}"})
        msg.append({"role": "user", "content": chat_prompt})

        print("msg", msg)
        response = client_gpt.chat.completions.create(
            model="gpt-4o",
            temperature=1,
            messages=msg,
            max_tokens=225,
        )
        fixed_prompt = response.choices[0].message.content
        return fixed_prompt

    async def send_long_message(self, channel, message):
        # if longer than 2000 characters, split it
        parts = [message[i:i + 2000] for i in range(0, len(message), 2000)]
        for part in parts:
            await channel.send(part)

    async def summarize_meeting_and_send(self):
        await self.files_saved_event.wait()  # Wait until all files are saved
        history = await process_and_summarize_json_files()  # Processing old data
        system_prompt = f"Summarize the following meeting details and extract actionable tasks for each participant in table format. Create response in language {language}!!! Not stack at {language} if they speak other language. Names of tools can be in english. It is transcribed text, so fix word that transcribed with errors. Be smart, don't write nonsense. Do it as if an ideal project manager would do it. Write down only tasks that are important or that have been explicitly stated. But don't write down meaningless tasks. Be local, don't write down obvious things. Don’t invent non-existent topics and facts, don’t hallucinate anything. If nothing was discussed, then write that the meeting did not take place."
        if history:
            messages = [history, {"role": "system", "content": system_prompt}]
        else:
            messages = [{"role": "system", "content": system_prompt}]
        chat_prompt = ""
        for audio_meta in self.recordings_metadata:
            filepath = audio_meta['audio_filename']
            username = audio_meta['username']
            timestamp = audio_meta['timestamp']
            print(f"Transcribing audio for {username}..." + filepath)
            transcribed = []
            for i in range(5):
                if i>=1 and i <= 3:
                    chat_prompt_temp = chat_prompt.split(",")
                    if len(chat_prompt_temp) > 6:
                        start = random.randrange(0, len(chat_prompt_temp)-3)
                        end = random.randrange(start, len(chat_prompt_temp)-1)
                        chat_prompt_temp = chat_prompt_temp[start:end]
                        chat_prompt_temp = ",".join(chat_prompt_temp)
                    else:
                        chat_prompt_temp = random.choice(["it is official hi tech team meeting, audio message from one of members", chat_prompt])
                elif i > 3:
                    chat_prompt_temp = ''
                else:
                    chat_prompt_temp = chat_prompt
                transcribed.append(self.transcribe_audio(filepath, chat_prompt_temp))
                print(f"Transcribed: {transcribed}")

            transcribed = self.work_around_gpt(str(transcribed), history=str(history))


            chat_prompt += f"{username}: {transcribed}\n"
            audio_meta['transcription'] = transcribed
            chat_prompt = await self.fix_chat_prompt(chat_prompt, history=str(history))
            message = {
                "role": "user",
                "content": f"{timestamp} | {username}: " + transcribed
            }
            messages.append(message)
        print("Messages:", messages)
        response = client_gpt.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=messages
        )
        summary = response.choices[0].message.content
        text_channel = discord.utils.get(self.voice_channel.guild.text_channels, name=general_channel_name)
        if text_channel:
            await self.send_long_message(text_channel, f"Meeting Summary:\n{summary}")
        else:
            print("Text channel 'general' not found")
        # save the metadata to a file
        json_filename = create_json_file(self.recordings_metadata)
        print(f'Metadata saved to {json_filename}')
        self.recordings_metadata = []


client = MyClient(intents=intents)
client.run(bot_token)
