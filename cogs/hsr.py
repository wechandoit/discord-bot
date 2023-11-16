# the Discord Python API
import nextcord as discord
from nextcord.ext import commands
from nextcord import application_command as app_commands

from mihomo import Language, MihomoAPI
from mihomo.models.v1 import StarrailInfoParsedV1

hsr_client = MihomoAPI(language=Language.EN)

async def setup(client: commands.Bot):
    client.add_cog(hsr(client))

class hsr(commands.Cog):

    def __init__(self, client: commands.Bot):
        self.client = client

    @app_commands.slash_command(name='hsr_stats', description='Get honkai star rail stats on a player')
    async def hsr_stats(self, interaction: discord.Interaction, uid):
        await interaction.response.defer()

        data: StarrailInfoParsedV1 = await hsr_client.fetch_user_v1(uid)
        name = data.player.name
        level = data.player.level
        achievements = data.player_details.achievements
        simulated_universe = data.player_details.simulated_universes
        pfp_url = hsr_client.get_icon_url(data.player.icon)
        char_count = data.player_details.characters

        char_list = data.characters
        party_list = []
        for character in char_list:
            party_list.append(character.name)

        embed=discord.Embed(description=f'Team: {", ".join(party_list)}\nCharacters Unlocked: {char_count}\nAchievements: {achievements}\nSimulated Universes Completed: {simulated_universe}', color=0x36ecc8)
        embed.set_author(name=f'{name} (Lvl {level})', icon_url=f'{pfp_url}')

        await interaction.followup.send(embed=embed)