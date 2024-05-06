import statistics
import sys
user_inputs = []

while True:
    user_input = input("Input a number: ")
    if user_input.isdigit():
        user_inputs.append(int(user_input))
    elif user_input == "Done":
        break
    else:
        print("You didn't input a number")

print(f"Number of user inputs: {len(user_inputs)}")
print(f"Average: {statistics.mean(user_inputs)}")
print(f"Min value: {min(user_inputs)}")
print(f"Max value: {max(user_inputs)}")

user_inputs.sort()

print(f"Sorted list: {user_inputs}")
sys.exit()