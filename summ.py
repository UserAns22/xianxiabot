import os, os.path
from nltk.tokenize import sent_tokenize


base_dir = './stories/'
output = './summs/{}.txt'

text = ''
for story in os.listdir(base_dir):
    story_output = output.format(story)
    story_text = ''
    story_dir = os.path.join(base_dir, story)
    for fname in os.listdir(story_dir):
        with open(os.path.join(story_dir, fname), 'r') as f:
            chapter_text = f.read()
            story_text += chapter_text
            text += chapter_text

    with open(story_output, 'w+') as o:
        o.write(story_text)
with open(output.format('combined'), 'w+') as o:
    o.write(text)


