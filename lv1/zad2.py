def get_grade_category(grade):
    if grade >= 0.9:
        return "A"
    elif grade >= 0.8:
        return "B"
    elif grade >= 0.7:
        return "C"
    elif grade >= 0.6:
        return "D"
    else:
        return "F"

try:
    while True:
        grade = float(input("Input a number between 0 and 1: "))
        if grade >= 0.0 and grade <= 1.0:
            print(f"Category: {get_grade_category(grade)}")
            break
except ValueError:
    print("Error. You didnt input a number")