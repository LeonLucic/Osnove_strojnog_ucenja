#prvi zad

sati=float((input("Unesi broj sati:")))

satnica=float(input("Unesi satnicu po satu:"))

def total_euro(sati,satnica):
    return (sati*satnica)

print(total_euro(sati,satnica))
print("Radni sati:",sati,"h")
print("eura/h:",satnica)
print("Ukupno",total_euro(sati,satnica),"eura")