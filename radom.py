import random

# Generate six random numbers between 1 and 38
numbers_1_to_38 = random.sample(range(1, 39), 6)

# Generate one random number between 1 and 7
number_1_to_7 = random.randint(1, 7)

print("Six random numbers between 1 and 38:", numbers_1_to_38)
print("One random number between 1 and 7:", number_1_to_7)