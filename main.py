def main():
    """
    The main function which is the entry point to the script.
    """
    print("Choose the option you want to run: ")
    print("1. DRL")
    print("2. WAN")
    print("3. TPJ")
    option = input("Enter option: ")

    if option == '1':
        drl()
    elif option == '2':
        wan()
    elif option == '3':
        tpj()
    else:
        print("Invalid option. Please choose a valid option.")

if __name__ == '__main__':
    main()