# the os module helps us access environment variables
# i.e., our API keys
import os
from dotenv import load_dotenv

# the Discord Python API
import nextcord as discord
from nextcord.ext import commands

# mongodb
from pymongo.mongo_client import MongoClient

# web scraping
from aiohttp import request
import requests

# import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# time
import time

# multithreading
import asyncio
import nest_asyncio
import gc

# string manipulation
import re
import nltk
from nltk.corpus import stopwords

# sentiment analysis
from textblob import TextBlob

nest_asyncio.apply()
sns.set_theme()

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix='$', intents=intents)

GUILD_IDS = (
    [int(guild_id) for guild_id in client.guilds]
    if client.guilds
    else discord.utils.MISSING
)

PROFANE_WORDS_URL = 'https://raw.githubusercontent.com/zacanger/profane-words/master/words.json'

profane_words = requests.get(PROFANE_WORDS_URL).json()
profane_words_regex = f'(?<![a-zA-Z0-9])({"|".join(profane_words)})(?![a-zA-Z0-9])'

nltk.download('punkt')
nltk.download('stopwords')
stopwords.words("english")[:10] # <-- import the english stopwords

### MONGODB SETUP

uri = os.getenv('MONGODB_URI')
mongo = MongoClient(uri)
try:
    mongo.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)      
    
@client.event
async def on_message(message):
    if f'{message.channel.id}' in mongo.list_database_names():
            
        line_df = pd.DataFrame({
            'Contents' : message.content.replace(',','').replace('\n', ' '),
            'Author' : message.author.id,
            'Timestamp' : message.created_at,
            'ContentsLength' : len(message.content)
        }, index=[0])
            
        # clean data and find the polarity/sentiment of the cleaned data
        line_df['Cleaned'] = line_df['Contents'].apply(lambda x: preprocess_text(x, remove_stopwords=True))
        line_df['Polarity'] = line_df['Cleaned'].apply(getPolarity)
        line_df['Sentiment'] = line_df['Polarity'].apply(getAnalysis)
        
        # save to mongodb
        db = mongo[f'{message.channel.id}']
        db.Iris.insert_many(line_df.to_dict("records"))
        
        # print(f"Logged message: {str(message.content)} into db {message.channel.id}")
                   
@client.slash_command(description='Load channel into db', guild_ids=GUILD_IDS)
async def log_channel(ctx):
    await ctx.response.defer()
    try:
        if f'{ctx.channel.id}' in mongo.list_database_names():
            await ctx.followup.send('This channel has been logged already!')
        else:
            messages, elapsed_time, contents, df = load_messages_to_df(ctx)
            combined_df = contents.join(df)
            
            # clean data and find the polarity/sentiment of the cleaned data
            combined_df['Cleaned'] = combined_df['Contents'].apply(lambda x: preprocess_text(x, remove_stopwords=True))
            combined_df['Polarity'] = combined_df['Cleaned'].apply(getPolarity)
            combined_df['Sentiment'] = combined_df['Polarity'].apply(getAnalysis)
            
            # save to mongodb
            db = mongo[f'{ctx.channel.id}']
            db.Iris.insert_many(combined_df.to_dict("records"))
            
            await ctx.followup.send(f'Elapsed time for {len(messages)} items: {elapsed_time}')
    except Exception as e:
        await ctx.followup.send(e)

@client.slash_command(description='Get channel stats', guild_ids=GUILD_IDS)
async def channel_stats(ctx, user: discord.Member = None):
    await ctx.response.defer()
    
    if f'{ctx.channel.id}' in mongo.list_database_names():
        if user == None:
            combined_df = pd.DataFrame(list(mongo[f'{ctx.channel.id}'].Iris.find()))
            combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            
            total_messages_sent = len(combined_df.index)
            total_characters_sent = int(combined_df['ContentsLength'].sum())
            print(f'{total_messages_sent} total messages sent')
            print(f'{total_characters_sent} total characters sent')
            print(f'{total_characters_sent // total_messages_sent} characters per message on average')
            
            count_by_week_ax = sns.histplot(data=combined_df['Timestamp'].dt.weekday, discrete=True)
            count_by_week_ax.set(title=f'Messages Sent Per Weekday in #{ctx.channel.name}', xlabel='Weekday', ylabel='# of Messages Sent')
            
            plt.savefig(f'plot-{ctx.channel.id}-weekly.png')
            plt.clf()
            
            hourly_ax = sns.histplot(data=combined_df['Timestamp'].dt.hour, discrete=True)
            hourly_ax.set(title=f'Messages Sent By Hour of Day in #{ctx.channel.name}', xlabel='Hour of Day', ylabel='# of Messages Sent')
            plt.savefig(f'plot-{ctx.channel.id}-hourly.png')
            plt.clf()
            
            print(f'{total_messages_sent} total messages sent')
            print(f'{total_characters_sent} total characters sent')
            print(f'{total_characters_sent // total_messages_sent} characters per message on average')
            
            await ctx.followup.send(f'#{ctx.channel.name} Stats:\n{total_messages_sent} total messages sent\n{total_characters_sent} total characters sent\n{total_characters_sent // total_messages_sent} characters per message on average', files=[discord.File(f'plot-{ctx.channel.id}-weekly.png'), discord.File(f'plot-{ctx.channel.id}-hourly.png')])
            
            os.remove(f'plot-{ctx.channel.id}-weekly.png')
            os.remove(f'plot-{ctx.channel.id}-hourly.png')
        else:
            combined_df = pd.DataFrame(list(mongo[f'{ctx.channel.id}'].Iris.find()))
            combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            filtered_df = combined_df.loc[combined_df['Author'] == user.id]
            
            total_messages_sent = len(filtered_df.index)
            total_characters_sent = int(filtered_df['ContentsLength'].sum())
            
            count_by_week_ax = sns.histplot(data=filtered_df['Timestamp'].dt.weekday, discrete=True)
            count_by_week_ax.set(title=f'Messages Sent Per Weekday in #{ctx.channel.name} from {user.name}', xlabel='Weekday', ylabel='# of Messages Sent')
            
            plt.savefig(f'plot-{ctx.channel.id}-weekly.png')
            plt.clf()
            
            hourly_ax = sns.histplot(data=filtered_df['Timestamp'].dt.hour, discrete=True)
            hourly_ax.set(title=f'Messages Sent By Hour of Day in #{ctx.channel.name} from {user.name}', xlabel='Hour of Day', ylabel='# of Messages Sent')
            plt.savefig(f'plot-{ctx.channel.id}-hourly.png')
            plt.clf()
            
            await ctx.followup.send(f'#{ctx.channel.name} Stats for {user.name}:\n{total_messages_sent} total messages sent\n{total_characters_sent} total characters sent\n{total_characters_sent // total_messages_sent} characters per message on average',files=[discord.File(f'plot-{ctx.channel.id}-weekly.png'), discord.File(f'plot-{ctx.channel.id}-hourly.png')])
            
            os.remove(f'plot-{ctx.channel.id}-weekly.png')
            os.remove(f'plot-{ctx.channel.id}-hourly.png')
            
    else:
        await ctx.followup.send('Please use the /log_channel command before using this command!')
    gc.collect()
    
@client.slash_command(description='Get channel stats about profanity', guild_ids=GUILD_IDS)
async def profanity_stats(ctx, user: discord.Member = None):
    await ctx.response.defer()
    try:
        if f'{ctx.channel.id}' in mongo.list_database_names():
            if user == None:
                combined_df = pd.DataFrame(list(mongo[f'{ctx.channel.id}'].Iris.find()))
                combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                profane_words_matches = combined_df['Cleaned'].str.extract(profane_words_regex)
                profane_words_matches = profane_words_matches[profane_words_matches.values != np.NaN].value_counts()
                ProfanitiesPage = 0
                profanities_page_size = 6

                pwm_top = profane_words_matches[ProfanitiesPage * profanities_page_size : (ProfanitiesPage + 1) * profanities_page_size]
                pwm_labels = [e[0] for e in pwm_top.index.to_list()]
                ax = sns.barplot(x=pwm_labels, y=pwm_top.values)
                ax.set(title='Profanity Use', ylabel='# of Messages With Profanity')
                
                plt.savefig(f'plot-{ctx.channel.id}-prof.png')
                plt.clf()
                
                await ctx.followup.send(file=discord.File(f'plot-{ctx.channel.id}-prof.png'))
                
                os.remove(f'plot-{ctx.channel.id}-prof.png')
            else:
                combined_df = pd.DataFrame(list(mongo[f'{ctx.channel.id}'].Iris.find()))
                combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                filtered_df = combined_df.loc[combined_df['Author'] == user.id]
                profane_words_matches = filtered_df['Cleaned'].str.extract(profane_words_regex)
                profane_words_matches = profane_words_matches[profane_words_matches.values != np.NaN].value_counts()
                ProfanitiesPage = 0
                profanities_page_size = 6

                pwm_top = profane_words_matches[ProfanitiesPage * profanities_page_size : (ProfanitiesPage + 1) * profanities_page_size]
                pwm_labels = [e[0] for e in pwm_top.index.to_list()]
                ax = sns.barplot(x=pwm_labels, y=pwm_top.values)
                ax.set(title=f'Profanity Use from {user.name}', ylabel=f'# of Messages With Profanity')
                
                plt.savefig(f'plot-{ctx.channel.id}-prof.png')
                plt.clf()
                
                await ctx.followup.send(file=discord.File(f'plot-{ctx.channel.id}-prof.png'))
                
                os.remove(f'plot-{ctx.channel.id}-prof.png')
        else:
            await ctx.followup.send('Please use the /log_channel command before using this command!')
        gc.collect()
    except Exception as e:
        await ctx.followup.send(e)

@client.slash_command(description='Get channel stats about emotions', guild_ids=GUILD_IDS)
async def emotion_stats(ctx, user: discord.Member = None):
    await ctx.response.defer()
    try:
        if f'{ctx.channel.id}' in mongo.list_database_names():
            if user == None:
                combined_df = pd.DataFrame(list(mongo[f'{ctx.channel.id}'].Iris.find()))
                
                # clean data
                combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                
                sentiment_over_time_by_month = group_and_transform_data(combined_df, pd.Series(combined_df['Timestamp'].dt.strftime('%Y-%m')), sentiment_table_to_positivity)
                ax = sentiment_over_time_by_month.plot()
                ax.set(title='Positivity Over Time', xlabel='Date', ylabel='Positivity %')
                for label in ax.get_xticklabels(which='major'):
                    label.set(rotation=25, horizontalalignment='right')
                    
                plt.savefig(f'plot-{ctx.channel.id}-emotions-monthly.png')
                plt.clf()
                
                sentiment_by_hour = group_and_transform_data(combined_df, pd.Series(combined_df['Timestamp'].dt.hour), sentiment_table_to_positivity)
                ax = sentiment_by_hour.plot()
                ax.set(title="Positivity By Hour of Day", xlabel="Hour of Day", ylabel="Positivity %")
                ax.set_xticks(range(0, 25, 4))
                
                plt.savefig(f'plot-{ctx.channel.id}-emotions-weekly.png')
                plt.clf()
                
                await ctx.followup.send(files=[discord.File(f'plot-{ctx.channel.id}-emotions-monthly.png'), discord.File(f'plot-{ctx.channel.id}-emotions-weekly.png')])
                
                os.remove(f'plot-{ctx.channel.id}-emotions-monthly.png')
                os.remove(f'plot-{ctx.channel.id}-emotions-weekly.png')
            else:
                combined_df = pd.DataFrame(list(mongo[f'{ctx.channel.id}'].Iris.find()))
                
                combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                filtered_df = combined_df.loc[combined_df['Author'] == user.id]
                
                sentiment_over_time_by_month = group_and_transform_data(filtered_df, pd.Series(filtered_df['Timestamp'].dt.strftime('%Y-%m')), sentiment_table_to_positivity)
                ax = sentiment_over_time_by_month.plot()
                ax.set(title=f'Positivity Over Time for {user.name}', xlabel='Date', ylabel='Positivity %')
                for label in ax.get_xticklabels(which='major'):
                    label.set(rotation=25, horizontalalignment='right')
                    
                plt.savefig(f'plot-{ctx.channel.id}-emotions-monthly.png')
                plt.clf()
                
                sentiment_by_hour = group_and_transform_data(filtered_df, pd.Series(filtered_df['Timestamp'].dt.hour), sentiment_table_to_positivity)
                ax = sentiment_by_hour.plot()
                ax.set(title=f"Positivity By Hour of Day for {user.name}", xlabel="Hour of Day", ylabel="Positivity %")
                ax.set_xticks(range(0, 25, 4))
                
                plt.savefig(f'plot-{ctx.channel.id}-emotions-weekly.png')
                plt.clf()
                
                await ctx.followup.send(files=[discord.File(f'plot-{ctx.channel.id}-emotions-monthly.png'), discord.File(f'plot-{ctx.channel.id}-emotions-weekly.png')])
                
                os.remove(f'plot-{ctx.channel.id}-emotions-monthly.png')
                os.remove(f'plot-{ctx.channel.id}-emotions-weekly.png')
        else:
            await ctx.followup.send('Please use the /log_channel command before using this command!')
        gc.collect()
    except Exception as e:
        await ctx.followup.send(e)

def preprocess_text(text: str, remove_stopwords: bool) -> str:
    # remove links
    text = str(text)
    if text and text.strip() != '':
        text = re.sub(r"http\S+", "", text)
        # remove special chars and numbers
        text = re.sub("[^A-Za-z]+", " ", text)
        # remove stopwords
        if remove_stopwords:
            # 1. tokenize
            tokens = nltk.word_tokenize(text)
            # 2. check if stopword
            tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
            # 3. join back together
            text = " ".join(tokens)
        # return text in lower case and stripped of whitespaces
        text = text.lower().strip()
        return text

def getPolarity(text):
   return TextBlob(str(text)).sentiment.polarity
        
def getAnalysis(score):
  if score < 0:
    return 'negative'
  elif score == 0:
    return 'neutral'
  else:
    return 'positive'
    
def sentiment_table_to_positivity(df):
  counts = df[df['Sentiment'] != 'neutral']['Sentiment'].value_counts()
  percentages = 100 * counts / counts.sum()
  pos = percentages.get('positive', default=0)
  return pos

def get_mentions_timeseries(df, keyword):
  keyword = keyword_match_wrapper(keyword)
  filtered_rows = df[df['Contents'].str.contains(keyword, case=False)]
  return group_and_transform_data(filtered_rows, pd.Series(filtered_rows['Timestamp'].dt.year), lambda df: len(df.index))

def keyword_match_wrapper(keyword: str) -> str:
  return f'(?<![a-zA-Z0-9]){keyword}(?![a-zA-Z0-9])'
    
async def get_message_list_from_iterator_async(iterator):
    return [item async for item in iterator if len(item.content) > 0]

def get_list_from_iterator(iterator):
    if asyncio.get_event_loop().is_running():
        loop = asyncio.get_event_loop()
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    start_time = time.time()
    result = loop.run_until_complete(get_message_list_from_iterator_async(iterator))
    elapsed_time = time.time() - start_time

    if loop is not asyncio.get_event_loop():
        loop.close()

    print(f'Elapsed time for {len(result)} items: {elapsed_time}')
    return result, elapsed_time

def group_and_transform_data(df, group_by, transformer):
  buckets = df.groupby(group_by)
  buckets = {b[0]: transformer(b[1]) for b in buckets}
  bucketed_df = pd.DataFrame.from_dict(buckets, orient='index')
  return bucketed_df

def load_messages_to_df(ctx):
    
    messages, elapsed_time  = get_list_from_iterator(ctx.channel.history(limit=None,oldest_first=True))
    message_contents = (x.content.replace('\n', ' ') for x in messages)
    message_author = (x.author.id for x in messages)
    message_timestamp = (x.created_at for x in messages)
    
    contents = pd.DataFrame({
    'Contents': message_contents,
    'Author': message_author
    })
    
    df = pd.DataFrame({
    'Timestamp': message_timestamp,
    'ContentsLength': contents['Contents'].str.len()
    })
    
    return messages, elapsed_time, contents, df

@client.event
async def on_ready():
    print('------')
    print(f'Logged in as {client.user.name}')
    print(client.user.id)
    print(f'In {len(client.guilds)} servers')
    print('------')
    
    await client.change_presence(activity=None)
    
async def change_status():
    await client.wait_until_ready()
    while not client.is_closed():
        await client.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name=f'to {len(client.guilds)} servers'))   
        await asyncio.sleep(10)

async def load():
    for filename in os.listdir('./cogs'):
        if filename.endswith('.py'):
            client.load_extension(f'cogs.{filename[:-3]}')

async def main():
    await load()
    client.loop.create_task(change_status())
    client.run(TOKEN)


if __name__ == '__main__':
    asyncio.run(main())
