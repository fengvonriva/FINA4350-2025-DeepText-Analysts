import os
import sys

def run_dictionary_method():
    print("\nDictionary-based Sentiment Analysis:")
    print("1. Run dictionary_sentiment_analysis.py (sentiment calculation)")
    print("2. Run price_and DICsentiment.py (price prediction with lag)")
    choice = input("Select an option (1/2, or q to go back): ").strip()
    if choice == '1':
        os.system(f"python ../analysis/dictionary_sentiment_analysis.py")
    elif choice == '2':
        os.system(f"python ../analysis/price_and DICsentiment.py")
    elif choice.lower() == 'q':
        return
    else:
        print("Invalid input.")

def run_nbow_method():
    print("\nNeural Bag of Words (NBoW) Sentiment Analysis:")
    print("1. Train NBoW model (NBoW.py)")
    print("2. Predict sentiment on Reddit comments (predict_sentiment.py)")
    print("3. Price prediction with NBoW sentiment (price_and_NBOWsentiment.py)")
    choice = input("Select an option (1/2/3, or q to go back): ").strip()
    if choice == '1':
        os.system(f"python ../analysis/NBoW.py")
    elif choice == '2':
        os.system(f"python ../analysis/predict_sentiment.py")
    elif choice == '3':
        os.system(f"python ../analysis/price_and_NBOWsentiment.py")
    elif choice.lower() == 'q':
        return
    else:
        print("Invalid input.")

def main():
    while True:
        print("\n==== Cryptocurrency Sentiment Analysis Main Menu ====")
        print("1. Dictionary-based Sentiment Analysis")
        print("2. Neural Bag of Words (NBoW) Sentiment Analysis")
        print("q. Quit")
        choice = input("Select an option (1/2/q): ").strip()
        if choice == '1':
            run_dictionary_method()
        elif choice == '2':
            run_nbow_method()
        elif choice.lower() == 'q':
            print("Exiting.")
            break
        else:
            print("Invalid input. Please try again.")

if __name__ == "__main__":
    main()
