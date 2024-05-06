def calculate_word_count(target_word):
    count = 0
    song_file = open('song.txt')
    lines = song_file.readlines()
    for line in lines:
        line = line.rstrip()
        words = line.split()
        for word in words:
            if word == target_word:
                count += 1

    song_file.close()
    return count

song_dict = {}

word_count_one_occurence = 0
song_file = open('song.txt')
lines = song_file.readlines()
for line in lines:
    line = line.rstrip()
    words = line.split()
    for word in words:
        song_dict[word] = calculate_word_count(target_word=word)
        if song_dict[word] == 1:
            word_count_one_occurence = word_count_one_occurence + 1 

song_file.close()
print(song_dict)
print(f"Word count with one occurence: {word_count_one_occurence}")