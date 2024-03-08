list = []

while True:
    a = input("Unesite broj: ")
    if a == "Done":
        break
    try:
        list.append(float(a))
    except:
        print("Unesite broj!")


print("Brojeva unijeto: ", len(list))
print("Srednja vrijednost: ", sum(list) / len(list))
print("Minimalna vrijednost: ", min(list))
print("Maksimalna vrijednost: ", max(list))
list.sort()
print("Sortirana lista: ", list)