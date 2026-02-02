x,y,z="wilson","helena", "cliford"
num=4
num2=str(4)
print(f"i thing i will love this sheet {num2}",end=" ")
print(type(num))

print(f"{x} is the oldest, {y} comes after, and {z} is last")

def greetElders(a):
    if a>25:
        print("good morning sir")
    else:
        print("we greet only elders")

greetElders(4)
thistuple = ("apple", "banana", "cherry")
print(thistuple[-3])