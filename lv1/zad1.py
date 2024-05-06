def total_euro(workingHours, payPerHour):
    return workingHours*payPerHour

workingHours = int(input("Working hours: "))
payPerHour = float(input("eur/h: "))
total = total_euro(workingHours, payPerHour)
print(f"Total: {total}")