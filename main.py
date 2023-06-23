# the os module helps us access environment variables
# i.e., our API keys
import os

# the Discord Python API
import nextcord as discord
from nextcord.ext import commands

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

matches = []
nest_asyncio.apply()
sns.set_theme()

intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix='$', intents=intents)

GUILD_IDS = (
    [int(guild_id) for guild_id in client.guilds]
    if client.guilds
    else discord.utils.MISSING
)

vlrggapi_url = 'https://vlrggapi.vercel.app/match/results'
vlrggapi_url_live = 'https://vlrggapi.vercel.app/match/upcoming'
PROFANE_WORDS_URL = 'https://raw.githubusercontent.com/zacanger/profane-words/master/words.json'

profane_words = requests.get(PROFANE_WORDS_URL).json()
profane_words_regex = f'(?<![a-zA-Z0-9])({"|".join(profane_words)})(?![a-zA-Z0-9])'

nltk.download('punkt')
nltk.download('stopwords')

stopwords.words("english")[:10] # <-- import the english stopwords
            

@client.event
async def on_ready():
    print('------')
    print(f'Logged in as {client.user.name}')
    print(client.user.id)
    print(f'In {len(client.guilds)} servers')
    print('------')
    
    await client.change_presence(activity=None)
    
@client.slash_command(description='Get information on recent valorant esports games', guild_ids=GUILD_IDS)
async def recent_matches(ctx, max: int = 5):
    
    max_search = 0
    
    async with request('GET', vlrggapi_url, headers={}) as response:
            if response.status == 200:
                    data = await response.json()
                    
                    max_search = len(data['data']['segments'])
                    if isinstance(max, int):
                        max_search = min(len(data['data']['segments']), int(max))

                    for item in data['data']['segments']:
                        if 'Champions Tour 2023' in item['tournament_name'] or 'North America' in item['tournament_name']:
                            team1 = item['team1']
                            team2 = item['team2']
                            score1 = int(item['score1'])
                            score2 = int(item['score2'])
                            tournament_icon = item['tournament_icon']
                            tournament_name = item['tournament_name'] + item['round_info']
                            
                            current_match = {
                                'team1': team1,
                                'team2': team2,
                                'score1': score1,
                                'score2': score2,
                                'tournament_icon': tournament_icon,
                                'tournament_name': tournament_name
                            }
                            
                            if not (current_match in matches):
                                matches.append(current_match)
                                
    for i in range(max_search):
        
        match = matches[i]
        
        team1 = match['team1']
        team2 = match['team2']
        score1 = match['score1']
        score2 = match['score2']
        tournament_icon = match['tournament_icon']
        tournament_name = match['tournament_name']
                    
        score = f'{score1}-{score2}'
        embed=discord.Embed(description=f'**Team 1:** {team1}\n**Team 2:** {team2}\n**Score:** {score}', color=0x36ecc8)
        embed.set_author(name=f'{tournament_name}', icon_url=f'{tournament_icon}')
                 
        await ctx.send(embed=embed)
        await asyncio.sleep(1)
                    
@client.slash_command(description='Get information on live valorant esports games', guild_ids=GUILD_IDS)
async def live_matches(ctx, max: int = 5):
    
    live_matches = []
    
    async with request('GET', vlrggapi_url_live, headers={}) as response:
            if response.status == 200:
                    data = await response.json()
                    count = 0

                    for item in data['data']['segments']:
                        if item['time_until_match'] == 'LIVE' and ('Champions Tour 2023' in item['tournament_name'] or 'North America' in item['tournament_name']):
                            team1 = item['team1']
                            team2 = item['team2']
                            
                            try:
                                score1 = int(item['score1'])
                            except ValueError:
                                score1 = 0
                            try:
                                score2 = int(item['score2'])
                            except ValueError:
                                score2 = 0
                            tournament_icon = item['tournament_icon']
                            tournament_name = item['tournament_name'] + ' ' + item['round_info']
                            
                            current_match = {
                                'team1': team1,
                                'team2': team2,
                                'score1': score1,
                                'score2': score2,
                                'tournament_icon': tournament_icon,
                                'tournament_name': tournament_name
                            }
                            
                            if not (current_match in live_matches):
                                live_matches.append(current_match)
                            
                            count += 1
    
    count = min(count, int(max))
    
    if len(live_matches) > 0:           
        for i in range(count):
            
            match = live_matches[i]
            
            team1 = match['team1']
            team2 = match['team2']
            score1 = match['score1']
            score2 = match['score2']
            tournament_icon = match['tournament_icon']
            tournament_name = match['tournament_name']
                        
            score = f'{score1}-{score2}'
            embed=discord.Embed(description=f'**Team 1:** {team1}\n**Team 2:** {team2}\n**Score:** {score}', color=0x36ecc8)
            embed.set_author(name=f'{tournament_name}', icon_url=f'{tournament_icon}')
                    
            await ctx.send(embed=embed)
            await asyncio.sleep(1)
    else:
        await ctx.send('There are no live matches right now.')

@client.slash_command(description='Get information on upcoming valorant esports games', guild_ids=GUILD_IDS)
async def upcoming_matches(ctx, max: int = 5):
    
    upcoming_matches = []
    
    async with request('GET', vlrggapi_url_live, headers={}) as response:
            if response.status == 200:
                    data = await response.json()
                    count = 0

                    for item in data['data']['segments']:
                        if item['time_until_match'] != 'LIVE' and ('Champions Tour 2023' in item['tournament_name'] or 'North America' in item['tournament_name']):
                            team1 = item['team1']
                            team2 = item['team2']
                            
                            try:
                                score1 = int(item['score1'])
                            except ValueError:
                                score1 = 0
                            try:
                                score2 = int(item['score2'])
                            except ValueError:
                                score2 = 0
                            tournament_icon = item['tournament_icon']
                            tournament_name = item['tournament_name'] + ' ' + item['round_info']
                            
                            current_match = {
                                'team1': team1,
                                'team2': team2,
                                'score1': score1,
                                'score2': score2,
                                'tournament_icon': tournament_icon,
                                'tournament_name': tournament_name
                            }
                            
                            if not (current_match in upcoming_matches):
                                upcoming_matches.append(current_match)
                            
                            count += 1
    
    count = min(count, int(max))
    
    if len(upcoming_matches) > 0:           
        for i in range(count):
            
            match = upcoming_matches[i]
            
            team1 = match['team1']
            team2 = match['team2']
            score1 = match['score1']
            score2 = match['score2']
            tournament_icon = match['tournament_icon']
            tournament_name = match['tournament_name']
                        
            score = f'{score1}-{score2}'
            embed=discord.Embed(description=f'**Team 1:** {team1}\n**Team 2:** {team2}\n**Score:** {score}', color=0x36ecc8)
            embed.set_author(name=f'{tournament_name}', icon_url=f'{tournament_icon}')
                    
            await ctx.send(embed=embed)
            await asyncio.sleep(1)
    else:
        await ctx.send('There are no live matches right now.')

@client.slash_command(description='Get valorant stats on a player', guild_ids=GUILD_IDS)
async def val_stats(ctx, player):
    
    
    player_name = player.split('#')[0]
    player_tag = player.split('#')[1]
    region, card_pfp_url = '', ''
    account_level = 0

    account_data_url = f'https://api.henrikdev.xyz/valorant/v1/account/{player_name}/{player_tag}'
    async with request('GET', account_data_url, headers={}) as response:
        if response.status == 200:
            data = await response.json()
            data = data['data']
            
            region = data['region'].upper()
            account_level = int(data['account_level'])
            card_pfp_url = data['card']['small']
            
            print(player, region)
    
    mmr_data_url = f'https://api.henrikdev.xyz/valorant/v1/mmr/{region}/{player_name}/{player_tag}'
    async with request('GET', mmr_data_url, headers={}) as response:
        if response.status == 200:
            data = await response.json()
            data = data['data']
            
            try:
                account_rank = data['currenttierpatched']
                account_rank_url = data['images']['small']
                account_rr = data['ranking_in_tier']
                last_game_rr = data['mmr_change_to_last_game']
                elo = data['elo']
            except Exception:
                account_rank = 'Not Ranked'
                account_rank_url = None
                account_rr = 0
                last_game_rr = 0
                elo = 0
                
            embed=discord.Embed(description=f'**Rank:** {account_rank}\n**RR:** {account_rr}/100\n**ELO:** {elo}\n**Level:** {account_level}\n**Last Game:** {last_game_rr}rr', color=0x36ecc8)
            
            if account_rank_url:
                embed.set_author(name=f'{player} ({region})', icon_url=f'{account_rank_url}')
            else:
                embed.set_author(name=f'{player} ({region})')
            
            embed.set_thumbnail(url=f'{card_pfp_url}')
            await ctx.send(embed=embed)
            
@client.event
async def on_message(message):
    if os.path.isfile(f'{message.channel.id}.csv'):
        with open(f'{message.channel.id}.csv', 'a+') as file:
            
            lines = len(pd.read_csv(f'{message.channel.id}.csv'))
            
            message_content = message.content.replace(',','').replace('\n', ' ')
            
            if len(message.content) > 0:
                file.write(f'{lines},{message_content},{message.author.id},{message.created_at},{len(message.content)}\n')
            file.close()
            
@client.slash_command(description='Load channel into db', guild_ids=GUILD_IDS)
async def log_channel(ctx):
    await ctx.response.defer()
    try:
        if os.path.isfile(f'{ctx.channel.id}.csv'):
            await ctx.followup.send('This channel has been logged already!')
        else:
            messages, elapsed_time, contents, df = load_messages_to_df(ctx)
            combined_df = contents.join(df)
            combined_df.to_csv(f'{ctx.channel.id}.csv')
            
            await ctx.followup.send(f'Elapsed time for {len(messages)} items: {elapsed_time}')
    except Exception as e:
        await ctx.followup.send(e)

@client.slash_command(description='Get channel stats', guild_ids=GUILD_IDS)
async def channel_stats(ctx, user: discord.Member = None):
    await ctx.response.defer()
    
    if os.path.isfile(f'{ctx.channel.id}.csv'):
        if user == None:
            combined_df = pd.read_csv(f'{ctx.channel.id}.csv')
            combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_convert('US/Eastern')
            
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
            combined_df = pd.read_csv(f'{ctx.channel.id}.csv')
            combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_convert('US/Eastern')
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
        if os.path.isfile(f'{ctx.channel.id}.csv'):
            if user == None:
                combined_df = pd.read_csv(f'{ctx.channel.id}.csv')
                combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_convert('US/Eastern')
                combined_df['Cleaned'] = combined_df['Contents'].apply(lambda x: preprocess_text(x, remove_stopwords=True))
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
                combined_df = pd.read_csv(f'{ctx.channel.id}.csv')
                combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_convert('US/Eastern')
                combined_df['Cleaned'] = combined_df['Contents'].apply(lambda x: preprocess_text(x, remove_stopwords=True))
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
        if os.path.isfile(f'{ctx.channel.id}.csv'):
            if user == None:
                combined_df = pd.read_csv(f'{ctx.channel.id}.csv')
                
                # clean data
                combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_convert('US/Eastern')
                combined_df['Cleaned'] = combined_df['Contents'].apply(lambda x: preprocess_text(x, remove_stopwords=True))
                combined_df['Polarity'] = combined_df['Cleaned'].apply(getPolarity)
                combined_df['Sentiment'] = combined_df['Polarity'].apply(getAnalysis)
                
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
                combined_df = pd.read_csv(f'{ctx.channel.id}.csv')
                
                combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_convert('US/Eastern')
                combined_df['Cleaned'] = combined_df['Contents'].apply(lambda x: preprocess_text(x, remove_stopwords=True))
                filtered_df = combined_df.loc[combined_df['Author'] == user.id]
                filtered_df['Polarity'] = filtered_df['Cleaned'].apply(getPolarity)
                filtered_df['Sentiment'] = filtered_df['Polarity'].apply(getAnalysis)
                
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

async def change_status():
    await client.wait_until_ready()
    while not client.is_closed():
        await client.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name=f'to {len(client.guilds)} servers'))   
        await asyncio.sleep(10)

def main():
    client.loop.create_task(change_status())
    client.run('MTA4Mzk0NjUzNjM3NzAwODI0MQ.GGTbtE.HabQRQ7-M7vApG-F01yXba489aKbhZhPrZOGH0')


if __name__ == '__main__':
  main()
