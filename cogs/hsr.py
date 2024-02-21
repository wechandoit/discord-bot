# the Discord Python API
import nextcord as discord
from nextcord.ext import commands
from nextcord import application_command as app_commands

import requests
import os
from starrailcard import honkaicard 

async def setup(client: commands.Bot):
    client.add_cog(hsr(client))
    
url = "https://starraillcard.up.railway.app/api/profile"
headers = {'Content-Type': 'application/json'}
    
# <:achievement_count:1207794228856365106> <:level_icon:1207794163349725214> <:char_icon:1207794164436180992> <:sim_universe:1207794161210626108> <:world_level:1207794162297217087>
        
async def main(uid):
    async with honkaicard.MiHoMoCard(template=2) as hmhm:
        files = []
        count = 0
        max_count = 4
        r = await hmhm.creat(uid)
        for key in r.card:
            if (count < max_count):
                card = key.card
                card.save(f"hsr-{key.id}.png")
                files.append(f"hsr-{key.id}.png")
                count += 1
            else:
                return files

class hsr(commands.Cog):

    def __init__(self, client: commands.Bot):
        self.client = client
        

    @app_commands.slash_command(name='hsr_stats', description='Get honkai star rail stats on a player')
    async def hsr_stats(self, interaction: discord.Interaction, uid):
        await interaction.response.defer()

        data = {
            "uid": f"{uid}",
            "lang": "en",
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("message") is None:
                print(f"HSR COG: Request successful for {uid}")
                files = await main(uid)
                embed=discord.Embed(description=f'<:level_icon:1207794163349725214> Level: {data["player"]["level"]}\n<:world_level:1207794162297217087> World Level: {data["player"]["world_level"]}\n\n<:achievement_count:1207794228856365106> Achievements: {data["player"]["space_info"]["achievement_count"]}\n<:char_icon:1207794164436180992> Characters Unlocked: {data["player"]["space_info"]["avatar_count"]}', color=0x36ecc8)
                embed.set_author(name=f'{data["player"]["nickname"]}'.upper())
                embed.set_thumbnail(url=f'{data["player"]["avatar"]["icon"]}')
                embed.add_field(name="Friends", value=f'{data["player"]["friend_count"]}', inline=True)
                embed.add_field(name="Light Cones", value=f'{data["player"]["space_info"]["light_cone_count"]}', inline=True)
                if files:
                    file = discord.File(files[0])
                    embed.set_image(url=f"attachment://{files[0]}")
                    await interaction.followup.send(file=file, embed=embed)
                    
                    for fa in files:
                        os.remove(fa)
                else:
                    await interaction.followup.send(embed=embed)
                
            else:
                print("HSR COG: Request failed")
                await interaction.followup.send('Make sure you put in the right uid!')
        else:
            await interaction.followup.send(f"Request failed with status code {response.status_code}")
        