# AI_ProjectManager

This bot records voice chats in a specified Discord voice channel, transcribes the recordings, summarizes the conversations, and sends the summary to a specified text channel. It serves as an AI-powered project manager to help keep track of important discussions and action items.

## Features
- Records audio from a Discord voice channel.
- Transcribes audio using OpenAI's Whisper model (either local or online).
- Summarizes conversations using OpenAI's GPT-4 model.
- Deletes audio files after processing to save space.

## Prerequisites
- Python 3.7+
- Discord Bot Token
- OpenAI API Key
- FFmpeg (for handling audio files)

## Installation

### Clone the Repository

Linux:
```bash
git clone https://github.com/stilletto/AI_ProjectManager.git
cd AI_ProjectManager
pip install -r requirements.txt
sudo apt update
sudo apt install ffmpeg
```
Windows
Download FFmpeg from https://ffmpeg.org/download.html.
Extract the downloaded file.
Add the bin folder to your system PATH.

Mac
```bash
brew install ffmpeg
```

### Configuration
Set your Discord Bot Token and OpenAI API Key in config.json
##How to Get API Keys
#Discord Bot Token

- Go to the Discord Developer Portal.

- Create a new application.

- Go to the "Bot" section and create a bot.

- Copy the token and add it to your config.json.

  
##OpenAI API Key

- Go to the OpenAI API Keys page.

- Generate a new API key.

- Copy the key and add it to your config.json.
