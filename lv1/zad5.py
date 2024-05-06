spam_count = 0
ham_count = 0
spam_count_ends_with_exclamation = 0

sms_file = open('SMSSpamCollection.txt')
lines = sms_file.readlines()
for line in lines:
    line = line.rstrip()
    if line.startswith('spam'):
        spam_count += 1
        if line.endswith('!'):
            spam_count_ends_with_exclamation += 1
    else:
        ham_count += 1

print(f"Spam count: {spam_count}")
print(f"Ham count: {ham_count}")
print(f"Spam count that ends with exclamation point: {spam_count_ends_with_exclamation}")