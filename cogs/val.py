# the Discord Python API
import nextcord as discord
from nextcord.ext import commands
from nextcord import application_command as app_commands

# web scraping
from aiohttp import request
import requests

async def setup(client: commands.Bot):
    client.add_cog(val(client))

class val(commands.Cog):
    def __init__(self, client: commands.Bot):
        self.client = client
    
    @app_commands.slash_command()
    async def test(self, interaction: discord.Interaction):
        await interaction.response.defer()
        await interaction.followup.send("Test")
    
    @app_commands.slash_command(description='Get valorant stats on a player')
    async def val_stats(self, interaction: discord.Interaction, player):
        await interaction.response.defer()
    
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
                except Exception:
                    account_rank = 'Not Ranked'
                    account_rank_url = None
                    account_rr = 0
                    last_game_rr = 0
                    
                embed=discord.Embed(description=f'**Rank:** {account_rank} ({account_rr}RR)\n**Level:** {account_level}\n**Last Game:** {last_game_rr}rr', color=0x36ecc8)
                
                if account_rank_url:
                    embed.set_author(name=f'{player} ({region})', icon_url=f'{account_rank_url}')
                else:
                    embed.set_author(name=f'{player} ({region})')
                
                embed.set_thumbnail(url=f'{card_pfp_url}')
                await interaction.followup.send(embed=embed)
                
    @app_commands.slash_command(description='Get a valorant players mmr history')
    async def mmr_history(self, interaction: discord.Interaction, player):
    
        await interaction.response.defer()
        
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
                
                embed.set_author(name=f'{player} ({region})')
                
                embed.set_thumbnail(url=f'{card_pfp_url}')
                await interaction.followup.send(embed=embed)
