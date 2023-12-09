# the os module helps us access environment variables
# i.e., our API keys
import os
from dotenv import load_dotenv

# the Discord Python API
import nextcord as discord
from nextcord.ext import commands
from nextcord import application_command as app_commands

# mongodb
from pymongo.mongo_client import MongoClient

# web scraping
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

# data analysis
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# sentiment analysis
from textblob import TextBlob

nest_asyncio.apply()
sns.set_theme()
load_dotenv()

PROFANE_WORDS_URL = 'https://raw.githubusercontent.com/zacanger/profane-words/master/words.json'
profane_words = requests.get(PROFANE_WORDS_URL).json()
profane_words_regex = f'(?<![a-zA-Z0-9])({"|".join(profane_words)})(?![a-zA-Z0-9])'

mention_regex = '<@(\d+)>'

stopwords.words("english")[:10] # <-- import the english stopwords  

async def setup(client: commands.Bot):
    client.add_cog(messages(client))

class messages(commands.Cog):
    
    def __init__(self, client: commands.Bot):
        self.client = client
        uri = os.getenv('MONGODB_URI')
        self.mongo = MongoClient(uri)
        try:
            self.mongo.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)
            
    # @title Start Wrapped '23 commands
    
    @app_commands.slash_command(name='my_wrapped', description='Get your server wrapped for 2023!')
    async def get_wrapped(self, interaction: discord.Interaction):
        await interaction.response.defer()
        if f'{interaction.channel.id}' in self.mongo.list_database_names():
            db = self.mongo[f'{interaction.channel.id}']
            if 'Wrapped' in db.list_collection_names():
                lookup_user_id = str(interaction.user.id)
                df = pd.DataFrame(list(db.Wrapped.find()))
                desc = ''
                
                user_dfs = {}
                for author in df['Author'].unique():
                    user_filtered_df = df.loc[df['Author'] == author]
                    user_dfs[str(author)] = user_filtered_df
                    
                # data analysis for the wrapped
                lookup_df = user_dfs[lookup_user_id]
                
                # top 5 used words
                most_used_word_dict = get_most_used_word_dict(lookup_df)
                
                top_n_words = sorted(most_used_word_dict, key=most_used_word_dict.get, reverse=True)[:5] # get top 5 words
                desc += '-'*40 + '\n'
                desc += get_top_n_words_formatted(df, user_dfs, top_n_words, most_used_word_dict, 'Top Word:', lookup_user_id, 5)
                    
                # top 5 used profanity
                
                user_profane_words_matches = lookup_df['Cleaned'].str.extract(profane_words_regex)
                user_profane_words_matches = user_profane_words_matches[user_profane_words_matches.values != np.NaN].value_counts().sort_values(ascending=False).to_frame().to_dict()
                user_profane_dict = user_profane_words_matches['count']
                top_n_words = [''.join(item) for item in sorted(user_profane_dict, key=user_profane_dict.get, reverse=True)[:5]]
                
                desc += '-'*40 + '\n'
                desc += get_top_n_words_formatted(df, user_dfs, top_n_words, most_used_word_dict, 'Top Profanity Word:', lookup_user_id, 5) # get top 5 profanity words
                
                # top 5 emotions
                emotion_messages = lookup_df['Emotion'][lookup_df['Emotion'] != 'neutral']
                user_emotion_data = emotion_messages.value_counts().sort_values(ascending=False).to_frame().to_dict()['count']
                top_n_emotions = [''.join(item) for item in sorted(user_emotion_data, key=user_emotion_data.get, reverse=True)[:5]]
                desc += '-'*40 + '\n'
                desc += f'Top Emotion: {top_n_emotions[0]} ({user_emotion_data[top_n_emotions[0]]} messages)\n'
                for i in range(1,5):
                    desc += f'{i+1}: {top_n_emotions[i]} ({user_emotion_data[top_n_emotions[i]]} messages)\n'
                
                # overall message count
                total_messages_sent = len(df.index)
                user_messages_sent = len(lookup_df.index)
                desc += '-'*40 + '\n'
                desc += f'You sent {user_messages_sent/total_messages_sent:.00%} of the messages in this channel\n'
                desc += f'You used {len(most_used_word_dict)} unique words last year!\n' # number of unique words
                
                embed=discord.Embed(title='Your 2023 Stats', description=desc, color=0x36ecc8)
                embed.set_author(name=f"@{interaction.user.name}'s #{interaction.channel.name} Wrapped", icon_url=interaction.user.avatar.url)
                embed.title
                
                await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send('This channel does not support this command! Please contact @wechandoit on discord if you think this is a mistake. He has to generate the data for these manually :<')
    
    # end Wrapped '23 commands     
        
    @commands.Cog.listener()
    async def on_message(self, message):
        if f'{message.channel.id}' in self.mongo.list_database_names():
                
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
            db = self.mongo[f'{message.channel.id}']
            db.Iris.insert_many(line_df.to_dict("records"))
            
            print(f"Logged message: {str(message.content)} into db {message.channel.id}")
                    
    @app_commands.slash_command(name='log_channel', description='Load channel into db')
    async def log_channel(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            if f'{interaction.channel.id}' in self.mongo.list_database_names():
                await interaction.followup.send('This channel has been logged already!')
            else:
                messages, elapsed_time, contents, df = load_messages_to_df(interaction)
                combined_df = contents.join(df)
                
                # clean data and find the polarity/sentiment of the cleaned data
                combined_df['Cleaned'] = combined_df['Contents'].apply(lambda x: preprocess_text(x, remove_stopwords=True))
                combined_df['Polarity'] = combined_df['Cleaned'].apply(getPolarity)
                combined_df['Sentiment'] = combined_df['Polarity'].apply(getAnalysis)
                
                # save to mongodb
                db = self.mongo[f'{interaction.channel.id}']
                db.Iris.insert_many(combined_df.to_dict("records"))
                
                await interaction.followup.send(f'Elapsed time for {len(messages)} items: {elapsed_time}')
        except Exception as e:
            await interaction.followup.send(e)

    @app_commands.slash_command(name='channel_stats', description='Get channel stats')
    async def channel_stats(self, interaction: discord.Interaction, user: discord.Member = None):
        await interaction.response.defer()
        
        if f'{interaction.channel.id}' in self.mongo.list_database_names():
            if user == None:
                combined_df = pd.DataFrame(list(self.mongo[f'{interaction.channel.id}'].Iris.find()))
                combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                
                total_messages_sent = len(combined_df.index)
                total_characters_sent = int(combined_df['ContentsLength'].sum())
                print(f'{total_messages_sent} total messages sent')
                print(f'{total_characters_sent} total characters sent')
                print(f'{total_characters_sent // total_messages_sent} characters per message on average')
                
                count_by_week_ax = sns.histplot(data=combined_df['Timestamp'].dt.weekday, discrete=True)
                count_by_week_ax.set(title=f'Messages Sent Per Weekday in #{interaction.channel.name}', xlabel='Weekday', ylabel='# of Messages Sent')
                
                plt.savefig(f'plot-{interaction.channel.id}-weekly.png')
                plt.clf()
                
                hourly_ax = sns.histplot(data=combined_df['Timestamp'].dt.hour, discrete=True)
                hourly_ax.set(title=f'Messages Sent By Hour of Day in #{interaction.channel.name}', xlabel='Hour of Day', ylabel='# of Messages Sent')
                plt.savefig(f'plot-{interaction.channel.id}-hourly.png')
                plt.clf()
                
                print(f'{total_messages_sent} total messages sent')
                print(f'{total_characters_sent} total characters sent')
                print(f'{total_characters_sent // total_messages_sent} characters per message on average')
                
                await interaction.followup.send(f'#{interaction.channel.name} Stats:\n{total_messages_sent} total messages sent\n{total_characters_sent} total characters sent\n{total_characters_sent // total_messages_sent} characters per message on average', files=[discord.File(f'plot-{interaction.channel.id}-weekly.png'), discord.File(f'plot-{interaction.channel.id}-hourly.png')])
                
                os.remove(f'plot-{interaction.channel.id}-weekly.png')
                os.remove(f'plot-{interaction.channel.id}-hourly.png')
            else:
                combined_df = pd.DataFrame(list(self.mongo[f'{interaction.channel.id}'].Iris.find()))
                combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                filtered_df = combined_df.loc[combined_df['Author'] == user.id]
                
                total_messages_sent = len(filtered_df.index)
                total_characters_sent = int(filtered_df['ContentsLength'].sum())
                
                count_by_week_ax = sns.histplot(data=filtered_df['Timestamp'].dt.weekday, discrete=True)
                count_by_week_ax.set(title=f'Messages Sent Per Weekday in #{interaction.channel.name} from {user.name}', xlabel='Weekday', ylabel='# of Messages Sent')
                
                plt.savefig(f'plot-{interaction.channel.id}-weekly.png')
                plt.clf()
                
                hourly_ax = sns.histplot(data=filtered_df['Timestamp'].dt.hour, discrete=True)
                hourly_ax.set(title=f'Messages Sent By Hour of Day in #{interaction.channel.name} from {user.name}', xlabel='Hour of Day', ylabel='# of Messages Sent')
                plt.savefig(f'plot-{interaction.channel.id}-hourly.png')
                plt.clf()
                
                await interaction.followup.send(f'#{interaction.channel.name} Stats for {user.name}:\n{total_messages_sent} total messages sent\n{total_characters_sent} total characters sent\n{total_characters_sent // total_messages_sent} characters per message on average',files=[discord.File(f'plot-{interaction.channel.id}-weekly.png'), discord.File(f'plot-{interaction.channel.id}-hourly.png')])
                
                os.remove(f'plot-{interaction.channel.id}-weekly.png')
                os.remove(f'plot-{interaction.channel.id}-hourly.png')
                
        else:
            await interaction.followup.send('Please use the /log_channel command before using this command!')
        gc.collect()
        
    @app_commands.slash_command(name='profanity_stats', description='Get channel stats about profanity')
    async def profanity_stats(self, interaction: discord.Interaction, user: discord.Member = None):
        await interaction.response.defer()
        try:
            if f'{interaction.channel.id}' in self.mongo.list_database_names():
                if user == None:
                    combined_df = pd.DataFrame(list(self.mongo[f'{interaction.channel.id}'].Iris.find()))
                    combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                    profane_words_matches = combined_df['Cleaned'].str.extract(profane_words_regex)
                    profane_words_matches = profane_words_matches[profane_words_matches.values != np.NaN].value_counts()
                    ProfanitiesPage = 0
                    profanities_page_size = 6

                    pwm_top = profane_words_matches[ProfanitiesPage * profanities_page_size : (ProfanitiesPage + 1) * profanities_page_size]
                    pwm_labels = [e[0] for e in pwm_top.index.to_list()]
                    ax = sns.barplot(x=pwm_labels, y=pwm_top.values)
                    ax.set(title='Profanity Use', ylabel='# of Messages With Profanity')
                    
                    plt.savefig(f'plot-{interaction.channel.id}-prof.png')
                    plt.clf()
                    
                    await interaction.followup.send(file=discord.File(f'plot-{interaction.channel.id}-prof.png'))
                    
                    os.remove(f'plot-{interaction.channel.id}-prof.png')
                else:
                    combined_df = pd.DataFrame(list(self.mongo[f'{interaction.channel.id}'].Iris.find()))
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
                    
                    plt.savefig(f'plot-{interaction.channel.id}-prof.png')
                    plt.clf()
                    
                    await interaction.followup.send(file=discord.File(f'plot-{interaction.channel.id}-prof.png'))
                    
                    os.remove(f'plot-{interaction.channel.id}-prof.png')
            else:
                await interaction.followup.send('Please use the /log_channel command before using this command!')
            gc.collect()
        except Exception as e:
            await interaction.followup.send(e)

    @app_commands.slash_command(name='emotion_stats', description='Get channel stats about emotions')
    async def emotion_stats(self, interaction: discord.Interaction, user: discord.Member = None):
        await interaction.response.defer()
        try:
            if f'{interaction.channel.id}' in self.mongo.list_database_names():
                if user == None:
                    combined_df = pd.DataFrame(list(self.mongo[f'{interaction.channel.id}'].Iris.find()))
                    
                    # clean data
                    combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                    
                    sentiment_over_time_by_month = group_and_transform_data(combined_df, pd.Series(combined_df['Timestamp'].dt.strftime('%Y-%m')), sentiment_table_to_positivity)
                    ax = sentiment_over_time_by_month.plot()
                    ax.set(title='Positivity Over Time', xlabel='Date', ylabel='Positivity %')
                    for label in ax.get_xticklabels(which='major'):
                        label.set(rotation=25, horizontalalignment='right')
                        
                    plt.savefig(f'plot-{interaction.channel.id}-emotions-monthly.png')
                    plt.clf()
                    
                    sentiment_by_hour = group_and_transform_data(combined_df, pd.Series(combined_df['Timestamp'].dt.hour), sentiment_table_to_positivity)
                    ax = sentiment_by_hour.plot()
                    ax.set(title="Positivity By Hour of Day", xlabel="Hour of Day", ylabel="Positivity %")
                    ax.set_xticks(range(0, 25, 4))
                    
                    plt.savefig(f'plot-{interaction.channel.id}-emotions-weekly.png')
                    plt.clf()
                    
                    await interaction.followup.send(files=[discord.File(f'plot-{interaction.channel.id}-emotions-monthly.png'), discord.File(f'plot-{interaction.channel.id}-emotions-weekly.png')])
                    
                    os.remove(f'plot-{interaction.channel.id}-emotions-monthly.png')
                    os.remove(f'plot-{interaction.channel.id}-emotions-weekly.png')
                else:
                    combined_df = pd.DataFrame(list(self.mongo[f'{interaction.channel.id}'].Iris.find()))
                    
                    combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                    filtered_df = combined_df.loc[combined_df['Author'] == user.id]
                    
                    sentiment_over_time_by_month = group_and_transform_data(filtered_df, pd.Series(filtered_df['Timestamp'].dt.strftime('%Y-%m')), sentiment_table_to_positivity)
                    ax = sentiment_over_time_by_month.plot()
                    ax.set(title=f'Positivity Over Time for {user.name}', xlabel='Date', ylabel='Positivity %')
                    for label in ax.get_xticklabels(which='major'):
                        label.set(rotation=25, horizontalalignment='right')
                        
                    plt.savefig(f'plot-{interaction.channel.id}-emotions-monthly.png')
                    plt.clf()
                    
                    sentiment_by_hour = group_and_transform_data(filtered_df, pd.Series(filtered_df['Timestamp'].dt.hour), sentiment_table_to_positivity)
                    ax = sentiment_by_hour.plot()
                    ax.set(title=f"Positivity By Hour of Day for {user.name}", xlabel="Hour of Day", ylabel="Positivity %")
                    ax.set_xticks(range(0, 25, 4))
                    
                    plt.savefig(f'plot-{interaction.channel.id}-emotions-weekly.png')
                    plt.clf()
                    
                    await interaction.followup.send(files=[discord.File(f'plot-{interaction.channel.id}-emotions-monthly.png'), discord.File(f'plot-{interaction.channel.id}-emotions-weekly.png')])
                    
                    os.remove(f'plot-{interaction.channel.id}-emotions-monthly.png')
                    os.remove(f'plot-{interaction.channel.id}-emotions-weekly.png')
            else:
                await interaction.followup.send('Please use the /log_channel command before using this command!')
            gc.collect()
        except Exception as e:
            await interaction.followup.send(e)

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

def get_top_n_words_formatted(df, user_dfs, top_n_words, most_used_word_dict, header, lookup_user_id, n):
    lookup_user_word_usage = [0, 0, 0, 0, 0]
    other_users_word_usage = [0, 0, 0, 0, 0]
    string = ''
    for i in range(n):
        for author in df['Author'].unique():
            if str(author) != lookup_user_id:
                other_df = user_dfs[str(author)]
                other_most_used_word_dict = get_most_used_word_dict(other_df)
                if top_n_words[i] in other_most_used_word_dict:
                    other_users_word_usage[i] += other_most_used_word_dict[top_n_words[i]]
            else:
                lookup_user_word_usage[i] = most_used_word_dict[top_n_words[i]]
    
    user_uses = lookup_user_word_usage[0]/(lookup_user_word_usage[0]+other_users_word_usage[0])
    string += f'{header} "{top_n_words[0]}" ({lookup_user_word_usage[0]} times used, {user_uses:.00%} of all uses)\n'
    for i in range(1,n):
        user_uses = lookup_user_word_usage[i]/(lookup_user_word_usage[i]+other_users_word_usage[i])
        string += f'{i+1}: "{top_n_words[i]}" ({lookup_user_word_usage[i]} times used, {user_uses:.00%} of all uses)\n'

    return string

def get_most_used_word_dict(df):
    try:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['Contents'])
        feature_names = vectorizer.get_feature_names_out()
        word_counts = X.sum(axis=0)
        word_count_dict = dict(zip(feature_names, word_counts.tolist()[0]))
        return word_count_dict
    except:
        return {}