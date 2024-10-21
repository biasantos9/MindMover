

def unify_names(Cond: str, Class: str) -> tuple:
    """
    Unify different representations of conditions and classes
    to a standard set of names.

    Parameters:
    - Cond (str): The input condition label.
    - Class (str): The input class label.

    Returns:
    - tuple: A tuple containing the standardized condition and class labels.
    """

    # Unify condition names
    if Cond.lower() == "inner" or Cond.lower() == "in":
        Cond = "Inner"
    elif Cond.lower() == "vis" or Cond.lower() == "visualized":
        Cond = "Vis"
    elif Cond.lower() == "pron" or Cond.lower() == "pronounced":
        Cond = "Pron"

    # Unify class names
    if Class.lower() == "all" or Class.lower() == "todo":
        Class = "All"
    elif Class.lower() == "up" or Class.lower() == "arriba":
        Class = "Up"
    elif Class.lower() == "down" or Class.lower() == "abajo":
        Class = "Down"
    elif Class.lower() == "right" or Class.lower() == "derecha":
        Class = "Right"
    elif Class.lower() == "left" or Class.lower() == "izquierda":
        Class = "Left"

    return Cond, Class



def sub_name(N_S: int) -> str:
    """
    Standardize subjects' names based on the input subject number.

    Parameters:
    - N_S (int): The subject number.

    Returns:
    - str: The standardized subject name.
    """
    if N_S < 10:
        Num_s = 'sub-0' + str(N_S)
    else:
        Num_s = 'sub-' + str(N_S)

    return Num_s
