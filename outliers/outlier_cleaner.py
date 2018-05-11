#!/C:/Python27


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    # to make a list with the same size of other lists
    errors = predictions
    for i in range(0, len(predictions)):
        errors[i] = abs(predictions[i] - net_worths[i])
    # using zip(...) will integrate multiple lists into one tuple
    myData = zip(net_worths, ages, errors)
    myData = sorted(myData, key=lambda myData: myData[2])
    # using zip(*...) will unpack one tuple into original lists
    net_worths, ages, errors = zip(*myData)
    cleaned_data = zip(ages[:81], net_worths[:81], errors[:81])
    # to check whether pack/unpack is right, we use next command (should result the same as non-outlier-removed model
    # cleaned_data = zip(ages, net_worths, errors)

    return cleaned_data
