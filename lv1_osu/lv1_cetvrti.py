
fhand = open("song.txt")

word_count = {}
unique_words=set()


for line in fhand:
    words=line.split()
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1

for word, count in word_count.items():
    if count == 1:
        unique_words.add(word)

print(len(unique_words))
print(word_count)