#def reverse(text):
    #return text[::-1]

def uppercase_and_reverse(text):
    #return reverse(text.upper())
    #return text.upper()[::-1]
    uppercase = text.upper()
    return uppercase[::-1]
print(uppercase_and_reverse("Biplob"))
