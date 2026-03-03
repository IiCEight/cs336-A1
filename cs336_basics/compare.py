import filecmp

if __name__ == "__main__":

    # with open("./data/train_token.bin", "rb") as f:
    #     lhs = f.read()

    # with open("./data/train_token_origin.bin", "rb") as f:
    #     rhs = f.read()

    # same = True
    # for i, (b1, b2) in enumerate(zip(lhs, rhs)):
    #     if b1 != b2:
    #         print(f"Byte mismatch at position {i}: {b1} != {b2}")
    #         same = False
    #         break

    # if same:
    #     print("The files are identical byte-for-byte.")
    # else:
    #     print("The files differ.")
    


    same = filecmp.cmp("./data/train_token.bin", "./data/train_token_origin.bin", shallow=False)

    if same:
        print("The files are identical byte-for-byte.")
    else:
        print("The files differ.")