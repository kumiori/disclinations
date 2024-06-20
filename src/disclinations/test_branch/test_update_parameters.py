def update_nested_dict(d, key, value):
    """
    Recursively traverses the dictionary d to find and update the key's value.
    
    Args:
    d (dict): The dictionary to traverse.
    key (str): The key to find and update.
    value: The new value to set for the key.
    
    Returns:
    bool: True if the key was found and updated, False otherwise.
    """
    if key in d:
        d[key] = value
        return True

    for k, v in d.items():
        if isinstance(v, dict):
            if update_nested_dict(v, key, value):
                return True
    
    return False

# Example usage
params = {
    'param1': 1,
    'nested': {
        'param2': 0.1,
        'param3': {
            'param4': 4
        }
    }
}

key_to_update = 'param4'
new_value = 10

print('Original params:')
print(params)

updated = update_nested_dict(params, key_to_update, new_value)
if updated:
    print(f"Updated {key_to_update} to {new_value}")
else:
    print(f"Key {key_to_update} not found")

print(f'Updated params? (True/False): {updated}')
print(params)
