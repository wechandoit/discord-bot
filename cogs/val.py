# the Discord Python API
import nextcord as discord
from nextcord.ext import commands
from nextcord import application_command as app_commands

# web scraping
from aiohttp import request

from collections import OrderedDict

async def setup(client: commands.Bot):
    client.add_cog(val(client))

# <:decrease:1208329278362755122> <:increase:1208329277464903700> <:stable:1208329279272652840>
def get_mmr_change_emoji(change: int) -> str:
    if change > 0:
        return '<:increase:1208329277464903700>'
    elif change < 0:
        return '<:decrease:1208329278362755122>'
    else:
        return '<:stable:1208329279272652840>'

class val(commands.Cog):
    def __init__(self, client: commands.Bot):
        self.client = client
    
    @app_commands.slash_command(name='val_stats', description='Get valorant stats on a player')
    async def val_stats(self, interaction: discord.Interaction, player):
        await interaction.response.defer()
    
        player_name = player.split('#')[0]
        player_tag = player.split('#')[1]
        region, card_pfp_url = '', ''
        account_level = 0

        has_card = True
        is_ranked = True

        account_data_url = f'https://api.henrikdev.xyz/valorant/v1/account/{player_name}/{player_tag}'
        async with request('GET', account_data_url, headers={}) as response:
            if response.status == 200:
                data = await response.json()
                data = data['data']
                
                region = data['region'].upper()
                account_level = int(data['account_level'])
                if 'card' in data:
                    card_pfp_url = data['card']['small']
                else:
                    has_card = False
                
                print(player, region)
            elif response.status == 404:
                print((f'Error 404: Player {player} does not exist'))
                await interaction.followup.send(f'Error: Player {player} does not exist!')
                return
            else:
                print(f"VAL STAT CMD 1: Error getting account info for {player}, error code {response.status}")
        
        mmr_data_url = f'https://api.henrikdev.xyz/valorant/v1/mmr/{region}/{player_name}/{player_tag}'
        async with request('GET', mmr_data_url, headers={}) as response:
            if response.status == 200:
                data = await response.json()
                data = data['data']
                
                try:
                    account_rank = data['currenttierpatched']
                    account_rank_url = data['images']['small']
                    account_rr = data['ranking_in_tier']
                except Exception:
                    account_rank = 'Not Ranked'
                    account_rank_url = None
                    account_rr = 0
                    is_ranked = False
                
                if not is_ranked:
                    embed=discord.Embed(description=f'', color=0x36ecc8)
                    embed.add_field(name='Rank', value=f'{account_rank} ({account_rr}RR)', inline=True)
                    embed.add_field(name='Level', value=account_level, inline=True)
                    
                    if account_rank_url:
                        embed.set_author(name=f'{player} ({region})', icon_url=f'{account_rank_url}')
                    else:
                        embed.set_author(name=f'{player} ({region})')
                    
                    if has_card:
                        embed.set_thumbnail(url=f'{card_pfp_url}')
                    await interaction.followup.send(embed=embed)
            else:
                print(f"VAL STAT CMD 2: Error getting account info for {player}, error code {response.status}")
        
        if is_ranked:
            mmr_data_url = f'https://api.henrikdev.xyz/valorant/v1/lifetime/mmr-history/{region}/{player_name}/{player_tag}'
            async with request('GET', mmr_data_url, headers={}) as response:
                if response.status == 200:
                    data = await response.json()
                    data = data['data']
                    
                    mmr_changes = []
                    match_info = []
                    count = 0
                    max_count = min(10, len(data))
                    while count < max_count:
                        match = data[count]
                        match_id = match['match_id']
                        mmr_change = match['last_mmr_change']
                        mmr_changes.append(int(mmr_change))
                        match_map = match['map']['name']
                        match_info.append(f'{mmr_change} ({match_map})')
                        rank_in_match = match['tier']['name']
                        date = match['date']
                        count += 1

                        print(match_id, match_map, mmr_change, date, rank_in_match)
                    
                    account_rank = data[0]['tier']['name']
                    account_rr = data[0]['ranking_in_tier']
                    
                    embed=discord.Embed(description=f'**{account_rank}** ({account_rr} / 100 RR)\n**Level**: {account_level}\n', color=0x36ecc8)
                    embed.add_field(name=f'RR history ({max_count} Games)',  value=", ".join(match_info), inline=False)
                    embed.add_field(name=f'RR Gain', value=sum(mmr_changes), inline=True)
                    embed.add_field(name=f'Average RR Gain', value=round(sum(mmr_changes)/len(mmr_changes), 2), inline=True)
                    
                    if account_rank_url:
                        embed.set_author(name=f'{player} ({region})', icon_url=f'{account_rank_url}')
                    else:
                        embed.set_author(name=f'{player} ({region})')
                    
                    if has_card:
                        embed.set_thumbnail(url=f'{card_pfp_url}')
                    await interaction.followup.send(embed=embed)
                else:
                    print(f"VAL STAT CMD 3: Error getting account info for {player}, error code {response.status}")
                
    @app_commands.slash_command(name='match_history', description='Get a valorant players match history')
    async def match_history(self, interaction: discord.Interaction, player):
    
        await interaction.response.defer()
        
        player_name = player.split('#')[0]
        player_tag = player.split('#')[1]
        region, card_pfp_url = '', ''
        player_puuid = None

        has_card = True

        account_data_url = f'https://api.henrikdev.xyz/valorant/v1/account/{player_name}/{player_tag}'
        async with request('GET', account_data_url, headers={}) as response:
            if response.status == 200:
                data = await response.json()
                data = data['data']
                
                region = data['region'].upper()
                player_puuid = data['puuid']
                if 'card' in data:
                    card_pfp_url = data['card']['small']
                else:
                    has_card = False
                
                print(player, region)
            elif response.status == 404:
                print((f'Error 404: Player {player} does not exist'))
                await interaction.followup.send(f'Error: Player {player} does not exist!')
                return
            else:
                print(f"MATCH HIST CMD 1: Error getting account info for {player}, error code {response.status}")
        
        mmr_data_url = f'https://api.henrikdev.xyz/valorant/v1/lifetime/matches/{region}/{player_name}/{player_tag}'
        async with request('GET', mmr_data_url, headers={}) as response:
            if response.status == 200:
                data = await response.json()
                data = data['data']
                match_count = min(10, len(data))

                # format match/player data
                map_list = []

                for i in range(match_count):
                    match = data[i]
                    m = val_map(match['meta']['map']['name'], match['meta']['mode'], str(match['teams']['red']) + "-" + str(match['teams']['blue']), str(match['teams']['blue']) + "-" + str(match['teams']['red']))

                    stats = match['stats']
                    team = stats['team']

                    match_player = val_player(player_name, player_tag, stats['puuid'], stats['character']['name'], stats['kills'], stats['deaths'], stats['assists'], team)
                    m.add_player(match_player, team, player_puuid)
                    map_list.append(m)

                # turn the data into a discord embed
                desc = ''
                for m in map_list:
                    desc += m.get_formatted_map() + "\n"
                
                embed=discord.Embed(description=desc, color=0x3c88eb)

                embed.set_author(name=f'{player} ({region}) | LAST {match_count} GAMES')

                if has_card:
                    embed.set_thumbnail(url=f'{card_pfp_url}')

                await interaction.followup.send(embed=embed)
            else:
                print(f"MATCH HIST CMD 2: Error getting account info for {player}, error code {response.status}")

    @app_commands.slash_command(name='comp_history', description='Get a valorant players competitive history')
    async def comp_history(self, interaction: discord.Interaction, player):
    
        await interaction.response.defer()
        
        player_name = player.split('#')[0]
        player_tag = player.split('#')[1]
        region, card_pfp_url = '', ''
        player_puuid = None
        has_card = True
        
        map_stats = OrderedDict()

        account_data_url = f'https://api.henrikdev.xyz/valorant/v1/account/{player_name}/{player_tag}'
        async with request('GET', account_data_url, headers={}) as response:
            if response.status == 200:
                data = await response.json()
                data = data['data']
                
                region = data['region'].upper()
                player_puuid = data['puuid']
                if 'card' in data:
                    card_pfp_url = data['card']['small']
                else:
                    has_card = False
                
                print(player, region)
            elif response.status == 404:
                print((f'Error 404: Player {player} does not exist'))
                await interaction.followup.send(f'Error: Player {player} does not exist!')
                return
            else:
                print(f"MATCH HIST CMD 1: Error getting account info for {player}, error code {response.status}")
        
        comp_data_url = f'https://api.henrikdev.xyz/valorant/v1/lifetime/matches/{region}/{player_name}/{player_tag}?mode=competitive'
        async with request('GET', comp_data_url, headers={}) as response:
            if response.status == 200:
                data = await response.json()
                data = data['data']
                match_count = min(10, len(data))

                for i in range(match_count):
                    match = data[i]
                    m = comp_map(match['meta']['map']['name'], str(match['teams']['red']) + "-" + str(match['teams']['blue']), str(match['teams']['blue']) + "-" + str(match['teams']['red']))

                    stats = match['stats']
                    team = stats['team']

                    match_player = comp_player(player_name, player_tag, stats['puuid'], stats['character']['name'], stats['kills'], stats['deaths'], stats['assists'], team, stats['shots']['head'], stats['shots']['body'], stats['shots']['leg'])
                    m.add_player(match_player, team, player_puuid)
                    map_stats[str(match['meta']['id'])] = m

        
        mmr_data_url = f'https://api.henrikdev.xyz/valorant/v1/lifetime/mmr-history/{region}/{player_name}/{player_tag}'
        async with request('GET', mmr_data_url, headers={}) as response:
            if response.status == 200:
                data = await response.json()
                data = data['data']
                
                mmr_changes = []
                match_info = []
                match_rr_dict = {}
                match_count = min(10, len(data))
                
                for i in range(match_count):
                    match = data[i]
                    match_id = match['match_id']
                    mmr_change = match['last_mmr_change']
                    mmr_changes.append(int(mmr_change))
                    match_map = match['map']['name']
                    match_info.append(f'{mmr_change} ({match_map})')
                    
                    match_rr_dict[str(match_id)] = mmr_change
                
                embed=discord.Embed(description='', color=0x3c88eb)

                embed.set_author(name=f'{player} ({region}) | LAST {len(map_stats.keys())} GAMES')
                
                for m in map_stats.keys():
                    formatted_map = map_stats[m].get_formatted_map()
                    if m in match_rr_dict:
                        embed.add_field(name=formatted_map[0] + f' ({get_mmr_change_emoji(match_rr_dict[m])} {match_rr_dict[m]} RR)', value=formatted_map[1], inline=False)
                    else:
                        embed.add_field(name=formatted_map[0], value=formatted_map[1], inline=False)

                if has_card:
                    embed.set_thumbnail(url=f'{card_pfp_url}')

                await interaction.followup.send(embed=embed)
        
            else:
                print(f"MATCH HIST CMD 2: Error getting account info for {player}, error code {response.status}")

class comp_player():
    def __init__(self, name, tag, puuid, character, kills, deaths, assists, team, hs, bs, ls):
        self.name = name
        self.tag = tag
        self.puuid = puuid
        self.character = character
        self.kills = kills
        self.deaths = deaths
        self.assists = assists
        self.kda_string = str(kills) + "-" + str(deaths) + "-" + str(assists)
        self.headshots = hs
        self.bodyshots = bs
        self.legshots = ls
        self.headshotpercent = float(hs)/(int(hs) + int(bs) + int(ls))
        self.team = team
    
    def get_full_tag(self):
        return self.name + "#" + self.tag
    
class comp_map():
    def __init__(self, map, red_score, blue_score):
        self.map = map
        self.red_score = red_score
        self.blue_score = blue_score
        self.red_players = []
        self.blue_players = []
        self.lookup_team = ''
        self.lookup_player = None
        
    
    def add_player(self, player: comp_player, team: str, lookup_puuid: str):
        if team == 'Red':
            self.red_players.append(player)
            if player.puuid == lookup_puuid:
                self.lookup_team = 'Red'
                self.lookup_player = player
        elif team == 'Blue':
            self.blue_players.append(player)
            if player.puuid == lookup_puuid:
                self.lookup_team = 'Blue'
                self.lookup_player = player

    def get_score(self):
        if self.lookup_team == 'Red':
            return self.red_score
        elif self.lookup_team == 'Blue':
            return self.blue_score
        else:
            return "chan fucked up :P"
        
    def get_formatted_map(self):
        formatted_map = []
        formatted_map.append(f'**{self.map} ({self.get_score()})**')
        if self.lookup_player == None:
            formatted_map.append(f'[Error getting stats... chan is bad]\n')
        else:
            formatted_map.append(f' - {self.lookup_player.character} (KDA: {self.lookup_player.kda_string} | HS%: {round(self.lookup_player.headshotpercent * 100, 1)}%)')
        return formatted_map

class val_player():
    def __init__(self, name, tag, puuid, character, kills, deaths, assists, team):
        self.name = name
        self.tag = tag
        self.puuid = puuid
        self.character = character
        self.kills = kills
        self.deaths = deaths
        self.assists = assists
        self.kda_string = str(kills) + "-" + str(deaths) + "-" + str(assists)
        self.team = team
    
    def get_full_tag(self):
        return self.name + "#" + self.tag

class val_map():
    def __init__(self, map, gamemode, red_score, blue_score):
        self.map = map
        self.gamemode = gamemode
        self.red_score = red_score
        self.blue_score = blue_score
        self.red_players = []
        self.blue_players = []
        self.lookup_team = ''
        self.lookup_player = None
        self.is_dm = self.gamemode == 'Deathmatch'
        
    
    def add_player(self, player: val_player, team: str, lookup_puuid: str):
        if not self.is_dm:
            if team == 'Red':
                self.red_players.append(player)
                if player.puuid == lookup_puuid:
                    self.lookup_team = 'Red'
                    self.lookup_player = player
            elif team == 'Blue':
                self.blue_players.append(player)
                if player.puuid == lookup_puuid:
                    self.lookup_team = 'Blue'
                    self.lookup_player = player
        else:
            if player.puuid == lookup_puuid:
                self.lookup_player = player

    def get_score(self):
        if not self.is_dm:
            if self.lookup_team == 'Red':
                return self.red_score
            elif self.lookup_team == 'Blue':
                return self.blue_score
            else:
                return "chan fucked up :P"
        
    def get_formatted_map(self):
        if not self.is_dm:
            string = f'**{self.map} | {self.gamemode} ({self.get_score()})**\n'
            if self.lookup_player == None:
                string += f'[Error getting stats... chan is bad]\n'
            else:
                string += f' - {self.lookup_player.character} (KDA: {self.lookup_player.kda_string})\n'
        else:
            string = f'**{self.map} | {self.gamemode}**\n'
            string += f' - {self.lookup_player.character} (KDA: {self.lookup_player.kda_string})\n'
        return string

