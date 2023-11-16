def ask(target_variable, target_value, known_evidence, bayesian_network):
    """
    Calculates the probability of a target variable being a certain value given some evidence.

    Parameters:
    target_variable (str): The variable for which the probability is being calculated.
    target_value (bool): The value (True or False) of the target variable.
    known_evidence (dict): A dictionary of known values for some variables.
    bayesian_network (BayesNet): The Bayesian Network.

    Returns:
    float: The calculated probability.
    """

    # Extend the known evidence to include the target variable
    extended_evidence = known_evidence.copy()
    #print(extended_evidence)
    extended_evidence[target_variable] = target_value
    #print(extended_evidence)

    def normalize(probabilities):
        """
        Normalizes a list of probabilities so they sum to 1.

        Parameters:
        probabilities (list): A list of probabilities.

        Returns:
        list: A list of normalized probabilities.
        """
        total_prob = sum(probabilities)
        return [prob / total_prob for prob in probabilities]

    def enumerate_all(variables, current_evidence):
        """
        Recursively calculates the joint probability by summing over all possible values
        of the variables not in the current evidence.

        Parameters:
        variables (list): List of variables to process.
        current_evidence (dict): The current state of evidence.

        Returns:
        float: The joint probability.
        """
        if not variables:
            return 1.0

        current_var = variables[0]
        #print(current_var)
        if current_var in current_evidence:
            # Use the known value from the evidence
            probability = bayesian_network.get_var(current_var).probability(current_evidence[current_var], current_evidence)
            return probability * enumerate_all(variables[1:], current_evidence)
        else:
            # Sum over all possible values (True, False) for the current variable
            total_prob = 0
            for var_value in [True, False]:
                current_evidence[current_var] = var_value
                total_prob += bayesian_network.get_var(current_var).probability(var_value, current_evidence) * enumerate_all(variables[1:], current_evidence)
                del current_evidence[current_var]
            return total_prob

    # Compute probabilities for both values (True and False) and normalize
    all_probabilities = [enumerate_all(bayesian_network.variable_names, dict(extended_evidence, **{target_variable: val})) for val in [True, False]]
    normalized_probabilities = normalize(all_probabilities)

    # Return the probability for the target value
    return normalized_probabilities[0] if target_value else normalized_probabilities[1]
