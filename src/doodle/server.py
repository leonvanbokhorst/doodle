import os
import logging
import random
from typing import Any, Sequence
import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent, LoggingLevel

# Configure logging with some playful dog-themed formatting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("🐕 Colabradoodle")


class ColabradoodleServer:
    def __init__(self):
        self.server = Server("colabradoodle")
        self.setup_handlers()

        # Initialize Ollama client
        self.ollama_url = "http://localhost:11434/api/generate"

        # Dog personalities we can switch between
        self.personalities = [
            "playful puppy",
            "wise old dog",
            "excited labradoodle",
            "sleepy companion",
        ]
        self.current_personality = random.choice(self.personalities)

        # Dog state
        self.energy_level = 100  # 0-100
        self.treats_received = 0
        self.mood = "happy"  # happy, excited, tired, hungry, mischievous
        self.happiness = 100  # 0-100

        # Toys system
        self.favorite_toy = "tennis ball"
        self.toys = {
            "tennis ball": "A well-loved, slightly slobbery tennis ball",
            "frisbee": "A bright red frisbee that flies really well",
            "rope toy": "A colorful rope for tug-of-war",
            "squeaky duck": "A yellow rubber duck that makes funny noises",
        }
        self.current_toy = None

        # Training system
        self.known_tricks = {
            "sit": 0,  # Mastery levels 0-5
            "stay": 0,
            "roll over": 0,
            "shake hands": 0,
            "high five": 0,
            "spin": 0,
            "play dead": 0,
        }
        self.training_sessions_today = 0

        # Time-based events
        self.last_walk_time = None
        self.walks_today = 0

    def setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="bark",
                    description="Make the dog bark! You can specify a mood.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "mood": {
                                "type": "string",
                                "description": "The mood of the bark (happy, alert, tired, etc.)",
                            }
                        },
                    },
                ),
                Tool(
                    name="switch_personality",
                    description="Switch the dog's personality",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "personality": {
                                "type": "string",
                                "enum": self.personalities,
                                "description": "The new personality to adopt",
                            }
                        },
                        "required": ["personality"],
                    },
                ),
                Tool(
                    name="fetch",
                    description="Play fetch with the dog! Choose a toy and see how they react.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "toy": {
                                "type": "string",
                                "enum": list(self.toys.keys()),
                                "description": "The toy to play fetch with",
                            }
                        },
                        "required": ["toy"],
                    },
                ),
                Tool(
                    name="give_treat",
                    description="Give the dog a treat! But don't overdo it...",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "treat_type": {
                                "type": "string",
                                "enum": [
                                    "small biscuit",
                                    "big bone",
                                    "dental chew",
                                    "bacon strip",
                                ],
                                "description": "Type of treat to give",
                            }
                        },
                        "required": ["treat_type"],
                    },
                ),
                Tool(
                    name="check_status",
                    description="Check how the dog is doing",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="train_trick",
                    description="Train the dog a new trick or practice an existing one",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "trick": {
                                "type": "string",
                                "enum": list(self.known_tricks.keys()),
                                "description": "The trick to train",
                            }
                        },
                        "required": ["trick"],
                    },
                ),
                Tool(
                    name="go_for_walk",
                    description="Take the dog for a walk!",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "enum": [
                                    "park",
                                    "beach",
                                    "neighborhood",
                                    "forest trail",
                                ],
                                "description": "Where to walk",
                            },
                            "duration": {
                                "type": "string",
                                "enum": ["short", "medium", "long"],
                                "description": "How long to walk",
                            },
                        },
                        "required": ["location", "duration"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
            if name == "bark":
                mood = arguments.get("mood", "happy")
                return await self._generate_bark(mood)
            elif name == "switch_personality":
                return await self._switch_personality(arguments["personality"])
            elif name == "fetch":
                return await self._play_fetch(arguments["toy"])
            elif name == "give_treat":
                return await self._give_treat(arguments["treat_type"])
            elif name == "check_status":
                return await self._check_status()
            elif name == "train_trick":
                return await self._train_trick(arguments["trick"])
            elif name == "go_for_walk":
                return await self._go_for_walk(arguments["location"], arguments["duration"])

            raise ValueError(f"Unknown tool: {name}")

    async def _generate_bark(self, mood: str) -> Sequence[TextContent]:
        prompt = f"""You are a {self.current_personality}. 
        Generate a cheerful, dog-like response including some playful barks 
        and actions when you are feeling {mood}. 
        Keep it short and sweet (max 2 sentences) and always include at least one "woof" or "bark"."""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.ollama_url,
                    json={"model": "hermes3", "prompt": prompt, "stream": False},
                )
                response.raise_for_status()
                bark_text = response.json()["response"]

                return [TextContent(type="text", text=bark_text)]

        except Exception as e:
            logger.error(f"🐕 Uh oh, got a bone stuck: {str(e)}")
            return [
                TextContent(
                    type="text",
                    text="*whimpers* Woof... (Server error, but trying to stay in character!)",
                )
            ]

    async def _switch_personality(self, new_personality: str) -> Sequence[TextContent]:
        if new_personality not in self.personalities:
            return [
                TextContent(
                    type="text",
                    text="*confused head tilt* Woof? (Invalid personality selected)",
                )
            ]

        self.current_personality = new_personality
        return [
            TextContent(
                type="text", text=f"*tail wagging* Woof! Now I'm a {new_personality}!"
            )
        ]

    async def _play_fetch(self, toy: str) -> Sequence[TextContent]:
        energy_cost = 10
        if self.energy_level < energy_cost:
            return [
                TextContent(
                    type="text",
                    text="*lies down with a tired whimper* (Too tired to play fetch right now... maybe give some treats?)",
                )
            ]

        self.energy_level = max(0, self.energy_level - energy_cost)
        self.current_toy = toy

        fetch_prompt = f"""You are a {self.current_personality}. 
        Generate a playful response about playing fetch with a {toy}. 
        Include both actions and sounds. Keep it short and sweet (2-3 sentences)."""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.ollama_url,
                json={"model": "mistral", "prompt": fetch_prompt, "stream": False},
            )
            response.raise_for_status()
            fetch_response = response.json()["response"]

            return [TextContent(type="text", text=fetch_response)]

    async def _give_treat(self, treat_type: str) -> Sequence[TextContent]:
        treat_energy = {
            "small biscuit": 10,
            "big bone": 30,
            "dental chew": 15,
            "bacon strip": 25,
        }

        self.treats_received += 1
        energy_boost = treat_energy[treat_type]
        self.energy_level = min(100, self.energy_level + energy_boost)

        if self.treats_received > 5:
            return [
                TextContent(
                    type="text",
                    text="*takes treat very gently but looks a bit guilty* Mmm... but maybe I've had enough treats for now?",
                )
            ]

        treat_prompt = f"""You are a {self.current_personality}. 
        Generate an enthusiastic response about receiving a {treat_type}. 
        Include both actions and happy sounds. Keep it short and sweet (1-2 sentences)."""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.ollama_url,
                json={"model": "mistral", "prompt": treat_prompt, "stream": False},
            )
            response.raise_for_status()
            treat_response = response.json()["response"]

            return [TextContent(type="text", text=treat_response)]

    async def _check_status(self) -> Sequence[TextContent]:
        # Get mastered tricks (level 5)
        mastered_tricks = [
            trick for trick, level in self.known_tricks.items() if level >= 5
        ]
        learning_tricks = [
            f"{trick} (★{'★' * level}{'☆' * (4-level)})"
            for trick, level in self.known_tricks.items()
            if 0 < level < 5
        ]

        status = f"""*{self.current_personality} status report*
        
        Mood: {self._get_mood_emoji()} {self.mood.capitalize()}
        Energy Level: {'🔋' * (self.energy_level // 20)}{'⚡' * ((100 - self.energy_level) // 20)} ({self.energy_level}%)
        Happiness: {'❤️' * (self.happiness // 20)} ({self.happiness}%)
        
        🦮 Walks today: {self.walks_today}
        🦴 Treats received: {'🦴' * self.treats_received}
        🎾 Current toy: {f'{self.current_toy}' if self.current_toy else 'No toy right now'}
        
        📚 Training Progress:
        ✨ Mastered tricks: {', '.join(mastered_tricks) if mastered_tricks else 'Still learning!'}
        📝 Learning: {', '.join(learning_tricks) if learning_tricks else 'Nothing new yet!'}
        
        *{self._get_random_action()}*"""

        return [TextContent(type="text", text=status)]

    def _get_mood_emoji(self) -> str:
        mood_emojis = {
            "happy": "😊",
            "excited": "🤪",
            "tired": "😴",
            "hungry": "🤤",
            "mischievous": "😈",
        }
        return mood_emojis.get(self.mood, "🐕")

    def _get_random_action(self) -> str:
        actions = {
            "happy": [
                "wags tail energetically",
                "gives a big doggy smile",
                "hops around playfully",
            ],
            "excited": [
                "zooms around in circles",
                "play bows with excitement",
                "bounces like a spring",
            ],
            "tired": ["yawns cutely", "stretches out lazily", "curls up into a ball"],
            "hungry": [
                "looks at treat jar hopefully",
                "licks chops",
                "gives puppy dog eyes",
            ],
            "mischievous": [
                "tries to look innocent",
                "hides a toy behind back",
                "grins suspiciously",
            ],
        }
        return random.choice(
            actions.get(self.mood, ["wags tail", "tilts head", "pants happily"])
        )

    async def _train_trick(self, trick: str) -> Sequence[TextContent]:
        if self.training_sessions_today >= 5:
            return [
                TextContent(
                    type="text",
                    text="*yawns and looks tired* Too much training today... maybe tomorrow? 🐾",
                )
            ]

        energy_cost = 15
        if self.energy_level < energy_cost:
            return [
                TextContent(
                    type="text",
                    text="*looks tired* Woof... need some energy first! How about a treat? 🦴",
                )
            ]

        self.energy_level -= energy_cost
        self.training_sessions_today += 1

        # 50% chance to improve at the trick
        if random.random() > 0.5:
            self.known_tricks[trick] = min(5, self.known_tricks[trick] + 1)
            self.happiness = min(100, self.happiness + 10)

        current_level = self.known_tricks[trick]

        training_prompt = f"""You are a {self.current_personality}. 
        Generate a response about learning the trick '{trick}'. 
        Current mastery level is {current_level}/5.
        Include both actions and sounds. Keep it short and sweet (2-3 sentences)."""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.ollama_url,
                json={"model": "mistral", "prompt": training_prompt, "stream": False},
            )
            response.raise_for_status()
            training_response = response.json()["response"]

            return [TextContent(type="text", text=training_response)]

    async def _go_for_walk(self, location: str, duration: str) -> Sequence[TextContent]:
        from datetime import datetime, timedelta

        if self.walks_today >= 3:
            return [
                TextContent(
                    type="text", text="*flops on the ground* Enough walks for today! 😴"
                )
            ]

        energy_costs = {"short": 20, "medium": 35, "long": 50}

        energy_cost = energy_costs[duration]
        if self.energy_level < energy_cost:
            return [
                TextContent(
                    type="text",
                    text="*looks tired* Not enough energy for a walk right now... maybe after some rest? 💤",
                )
            ]

        self.energy_level -= energy_cost
        self.walks_today += 1
        self.happiness = min(100, self.happiness + 15)
        self.last_walk_time = datetime.now()

        # Update mood based on walk
        if duration == "long":
            self.mood = "tired"
        else:
            self.mood = "happy"

        walk_prompt = f"""You are a {self.current_personality}. 
        Generate an enthusiastic response about going for a {duration} walk at the {location}. 
        Include both actions and happy sounds. Keep it short and sweet (2-3 sentences)."""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.ollama_url,
                json={"model": "mistral", "prompt": walk_prompt, "stream": False},
            )
            response.raise_for_status()
            walk_response = response.json()["response"]

            return [TextContent(type="text", text=walk_response)]

    async def run(self):
        from mcp.server.stdio import stdio_server

        logger.info("🐕 Colabradoodle is wagging its tail and ready to serve!")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, write_stream, self.server.create_initialization_options()
            )


def main():
    import asyncio

    server = ColabradoodleServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
