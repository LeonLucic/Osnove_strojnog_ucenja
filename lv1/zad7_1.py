"""Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo
1912. godine. Upoznajte se s datasetom i dodajte programski kod u skriptu pomo´cu kojeg možete
odgovoriti na sljede´ca pitanja:"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("titanic.csv")

# a) za koliko žena postoje podaci u ovom skupu podataka

female=data[data["Sex"]=="female"]
print(f"Broj žena je: {len(female)} ")

# b) Koliki postotak osoba nije prezivio potonuce broda

survived=data[data["Survived"]==0]
print(f"Broj preživjelih osoba je: {(len(survived)/len(data))*100}")

# c) Pomo´cu stupˇcastog dijagrama prikažite postotke preživjelih muškaraca (zelena boja) i žena
#(žuta boja). Dodajte nazive osi i naziv dijagrama. Komentirajte korelaciju spola i postotka
#preživljavanja.

male=data[data["Sex"]=="male"]
prezivjeli_m=male[male["Survived"]==1]
postotak_m=(len(prezivjeli_m)/len(male))*100

female=data[data["Sex"]=="female"]
prezivjele_f=female[female["Survived"]==1]
postotak_f=(len(prezivjele_f)/len(female))*100

spolovi=["Zene", "Muskarci"]
postotci=[postotak_f, postotak_m]
boje=["yellow", "green"]

plt.bar(spolovi, postotci, color=boje)
plt.xlabel("spol")
plt.ylabel("postotak")
plt.show()

# d)Kolika je prosjeˇcna dob svih preživjelih žena, a kolika je prosjeˇcna dob svih preživjelih
# muškaraca?

print(f"Prosjecna dob prezivjelih zena je: {prezivjele_f['Age'].mean()}")
print(f"Prosjecna dob prezivjelih muskaraca je: {prezivjeli_m['Age'].mean()}")

# e)Koliko godina ima najstariji preživjeli muškarac u svakoj od klasa? Komentirajte.

prezivjeli_m1=prezivjeli_m[prezivjeli_m["Pclass"]==1]
print(f"Najstariji prezivjeli muskarac u prvoj klasi ima {prezivjeli_m1['Age'].max()}")

prezivjeli_m2=prezivjeli_m[prezivjeli_m["Pclass"]==2]
print(f"Najstariji prezivjeli muskarac u drugoj klasi ima {prezivjeli_m2['Age'].max()}")

prezivjeli_m3=prezivjeli_m[prezivjeli_m["Pclass"]==3]
print(f"Najstariji prezivjeli muskarac u trecoj klasi ima {prezivjeli_m3['Age'].max()}")
