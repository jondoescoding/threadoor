# Midjourney - Prompt Length

Created time: May 9, 2023 6:53 PM
Last edited time: May 9, 2023 11:15 PM
Owner: Jon
Verification: Verified

---

# Understanding Prompts In Midjourney

## The concept of “Zone Of Influence”

- In midjourney versions 1-4, niji and test all have a system where the more words there will be a less likely chance of those words playing a role in the final generation of the image.
    - You can think of it like this where the closer you get to **60+ words**, the amount of influence each word would have over the image to be generated will be extremely low.
    - So the idea is to think of this as a downward facing slope where
        - more words = less likely to appear in image
    - What has been observed is that ***before*** we even each 60 words the amount of influence the words has almost negligible influence over the words
    - It can be thought of like this:
        - Words 1-5: Very influential, will probably show up on first iteration of images.
        - Words 6-20: Still in play, you may get lucky, may take 2-3 image rerolling.
        - Words 21-40: Somewhat still in play, possibly take more tries and will look as if it is being ignored
        - Words 40+: May take many more tries, will likely appear ignored.
        - Words 60+: Even more likely to appear ignored.
        - Words 70+: Just no.
    - In the version 5 series of MJ the influence of the words isn’t determined by **WHERE** the words are in relation to the start of the prompt but instead it is determined by its *strength*

## Combination & Permutation

- A permutation refers to a type of mathematical technique which determines the number of possible arrangements of a set.
    - This set can be a number, arbitrary objects etc.
- As this relates to prompting more words means that the resulting image may have a far less likely chance to see everything that is being expressed
    - As an example I was making art to post on my [twitter](http://twitter.com/jondidathing):
        - A young, tall and dark elf sitting on a lush grassy meadow brimming with both flora and fauna. A single large black dire wolf is beside the elf. The dire wolf is lying down beside, the elf eyes closed. The sky is bright blue, moon in clear view, with an asteroid belt of rocks. Theme: summer time in a fantasy world.
            
        - It got everything up to the asteroid belt correct.
    - What’s happening under the hood here is that there for each word there are a number of possible combinations which the image generation could take.
        - As a super example, 3 words = 8 possible combinations of what I want expressed in my art
        - 9 words = 512 possible combination
        - In my image above? 60 words = A whole hell of a lot of combinations
    - In version 5 of MJ the permutations concept is akin to a game of football where there are multiple coaches arguing what the players should do
    - In V4 more words means that they are getting lost in a vast ocean of permutations and you’ll never be able to see how they influence the image

## Word Salad

- You’ll often see prompts written out in a series of words almost as if a foreigner is trying to speak our native tongue
    - Good example is: banana on table, hyper realistic, ue4, dynamic angle, dusty dark room
- Prompts like these would work in MJ4 and under but in MJ5 there is some natural language processing at play so its best to play around writing how we normally do in our day to day.

## Delimiters

- How can I combat this issue and get my prompts to show as much as possible?
    - You can use a delimiter or ::
    - Using a delimiter (in MJ5) we can place emphasis or strength on a specific segment of our prompt
        - Using my prompt before as an example there is no use of :: so as the prompt continues in length the likely chance of seeing an asteroid belt in the sky diminishes
        - Adjusting the prompts length so there is less words and adding delimiters creates a whole new prompt which shows us the “rocks” surrounding the moon