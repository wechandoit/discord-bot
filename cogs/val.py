# the Discord Python API
import nextcord as discord
from nextcord.ext import commands
from nextcord import application_command as app_commands

# web scraping
from aiohttp import request

async def setup(client: commands.Bot):
    client.add_cog(val(client))

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
                    last_game_rr = data['mmr_change_to_last_game']
                except Exception:
                    account_rank = 'Not Ranked'
                    account_rank_url = None
                    account_rr = 0
                    last_game_rr = 0
                    is_ranked = False
                
                if not is_ranked:
                    embed=discord.Embed(description=f'**Rank:** {account_rank} ({account_rr}RR)\n**Level:** {account_level}\n', color=0x36ecc8)
                    
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
                        date = match['date']
                        count += 1

                        print(match_id, match_map, mmr_change, date)
                    
                    account_rank = data[0]['tier']['name']
                    account_rr = data[0]['ranking_in_tier']
                    
                    embed=discord.Embed(description=f'**Rank:** {account_rank} ({account_rr}RR)\n**Level:** {account_level}\n**Average RR Gain in {max_count} Games:** {sum(mmr_changes)/len(mmr_changes)}\n**RR history:** {", ".join(match_info)}', color=0x36ecc8)
                    
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
                
                embed=discord.Embed(description=desc)

                embed.set_author(name=f'{player} ({region}) | LAST {match_count} GAMES')

                if has_card:
                        embed.set_thumbnail(url=f'{card_pfp_url}')

                await interaction.followup.send(embed=embed)
            else:
                print(f"MATCH HIST CMD 2: Error getting account info for {player}, error code {response.status}")

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

