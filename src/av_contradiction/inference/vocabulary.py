"""Concepts for the dashboard's zero-shot text tags: what can be seen (scored with CLIP) and what can
be heard (scored with CLAP). A heard concept is looked up in the frames by its name, so sound concepts
are phrased by their source where possible ("a dog barking", not "barking")."""

VISUAL_CONCEPTS = [
    # people
    "a man", "a woman", "a child", "a baby", "a crowd of people", "people talking", "a person speaking to the camera",
    "a news presenter", "a teacher in a classroom", "a person on the phone", "a person singing", "people dancing",
    "people laughing", "people clapping", "a person applying makeup", "a close-up of a face", "a close-up of an eye",
    "a person cooking", "a person eating", "a person exercising", "a person sleeping", "a person walking",
    # music
    "a guitar", "an electric guitar", "a piano", "a violin", "drums", "a trumpet", "a saxophone", "a flute",
    "a cello", "a harp", "an accordion", "a band on stage", "an orchestra", "a DJ", "a microphone",
    # animals
    "a dog", "a cat", "birds", "a rooster", "a chicken", "a cow", "a horse", "a sheep", "a goat", "a pig",
    "a duck", "a frog", "insects", "a lion", "an elephant", "a monkey", "a whale", "a fish", "bees",
    # vehicles and machines
    "a car", "car traffic", "a motorcycle", "a truck", "a bus", "a train", "an airplane", "a helicopter",
    "a boat", "a bicycle", "an ambulance", "a police car", "a fire truck", "a lawn mower", "a chainsaw",
    "a drill", "construction work", "a vacuum cleaner", "a washing machine", "a sewing machine",
    "a computer", "a computer screen", "a phone", "a television", "a clock", "a camera",
    # objects and close-ups
    "food on a plate", "a cake", "drinks", "a product on a table", "makeup products", "clothes", "shoes",
    "jewellery", "a toy", "a book", "text on a screen", "a video game", "a cartoon", "a map", "money",
    # nature and weather
    "rain", "a storm", "snow", "ocean waves", "a river", "a waterfall", "a fire", "fireworks", "a forest",
    "the beach", "a mountain", "a desert", "a garden", "flowers", "the sky", "a sunset",
    # sport
    "a football match", "a basketball game", "tennis", "a boxing match", "skateboarding", "swimming",
    "a gym", "a car race", "people running",
    # places
    "a busy street", "a quiet library", "an office", "a classroom", "a restaurant", "a shop", "a kitchen",
    "a bathroom", "a living room", "a bedroom", "a car interior", "a train station", "an airport",
    "a stadium", "a concert", "a park", "a farm", "a church", "a hospital", "a factory", "a city at night",
    "a village", "a swimming pool",
]

SOUND_CONCEPTS = [
    # voices
    "a man speaking", "a woman speaking", "a child speaking", "a narrator speaking", "people talking",
    "a crowd", "a person singing", "a choir", "a baby crying", "children playing", "people laughing",
    "people clapping", "people cheering", "a person shouting", "a person whispering", "a person coughing",
    "footsteps",
    # music
    "background music", "a guitar", "an electric guitar", "a piano", "a violin", "drums", "a trumpet",
    "a saxophone", "a flute", "a cello", "a harp", "an accordion", "a band playing", "an orchestra",
    "electronic dance music", "a church organ",
    # animals
    "a dog barking", "a cat meowing", "birds singing", "a rooster crowing", "chickens", "a cow mooing",
    "a horse", "sheep", "a pig", "ducks", "frogs", "insects buzzing", "a lion roaring", "an elephant",
    "monkeys", "bees",
    # vehicles and machines
    "a car engine", "car traffic", "a car horn", "a motorcycle", "a truck", "a bus", "a train", "an airplane",
    "a helicopter", "a boat engine", "a bicycle bell", "a siren", "a lawn mower", "a chainsaw", "a drill",
    "construction work", "a vacuum cleaner", "a washing machine", "a sewing machine", "typing on a keyboard",
    "a clock ticking", "a doorbell", "a phone ringing", "an alarm", "a camera shutter",
    # nature and weather
    "rain", "thunder", "wind", "ocean waves", "a river", "a waterfall", "a fire crackling", "fireworks",
    # daily life
    "cooking and frying food", "water running from a tap", "dishes clattering", "a door closing", "eating",
    "pouring a drink", "a toilet flushing", "a shower", "a hair dryer",
    # sport and action
    "a football crowd", "a ball bouncing", "a referee whistle", "gunshots", "an explosion", "a video game",
    "silence",
]

# Sounds whose source is normally not on screen; they are not looked up in the frames.
OFF_SCREEN_SOUNDS = frozenset({"background music", "a narrator speaking", "electronic dance music", "silence"})

VISUAL_TEMPLATE = "a photo of {}"
AUDIO_TEMPLATE = "the sound of {}"

# Common English words that are not useful transcript keywords.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are aren't as at be because been before being below between
both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each even ever every few
for from further get gets getting go goes going gonna got had hadn't has hasn't have haven't having he he'd he'll he's her
here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself just
know let's like look me more most much must mustn't my myself no nor not now of off oh ok okay on once one only or other
ought our ours ourselves out over own really right said same say says see she she'd she'll she's should shouldn't so some
such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've
thing things think this those though through to too uh um under until up us very want was wasn't way we we'd we'll we're
we've well were weren't what what's when when's where where's which while who who's whom why why's will with won't would
wouldn't yeah yes you you'd you'll you're you've your yours yourself yourselves take give make put get use going
""".split())
