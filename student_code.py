def ask(var, value, evidence, bn):
    """Computes P(var=value | evidence), given a Bayes net bn"""
    probabilities = []
    for var_value in [True, False]:
        extended_evidence = evidence.copy()
        extended_evidence[var] = var_value
        probabilities.append(joint_prob(bn.variable_names, extended_evidence, bn))

    normalised_probabilities = normalisation(probabilities)

    if value:
        return normalised_probabilities[0]
    else:
        return normalised_probabilities[1]
 
    
def normalisation(probabilities):
    """Normalises a probability distribution so that it sums to 1"""
    total_prob = sum(probabilities)
    return [prob / total_prob for prob in probabilities]


def joint_prob(variables, current_evidence, bn):
    """Computes the joint probability of a variable being a particular value
    given some evidence, in a Bayes net bn"""
    if not variables:
        return 1.0
    
    current_var = variables[0]
    if current_var in current_evidence:
        probability = bn.get_var(current_var).probability(current_evidence[current_var], current_evidence)
        joint_probability = probability * joint_prob(variables[1:], current_evidence, bn)
        return joint_probability
    else:
        # Sum over all possible values (True, False) for the current variable 
        total_prob = 0
        for var_value in [True, False]:
            current_evidence[current_var] = var_value
            total_prob += bn.get_var(current_var).probability(var_value, current_evidence) * joint_prob(variables[1:], current_evidence, bn)
            del current_evidence[current_var]
        return total_prob

    
