#Recognize innacurate pose
def features_analysis(features):
    '''Determine which corrections are needed from features'''
    inaccurates = []
    
    if features['norm_dist__left_hip__left_hand'] > 0.05:
        inaccurates.append('Left hand is not on belt')

    if features['norm_dist__right_hip__right_hand'] > 0.05:
        inaccurates.append('Right hand is not on belt')

    if features['norm_dist__left_knee__right_foot'] < 0.03:
        if features['angle__left_thigh__left_calf'] < 170:
            inaccurates.append('Left leg is not straight')
        elif features['angle__left_upper__left_lower'] < 160:
            inaccurates.append('Body is not straight')
        
    elif features['norm_dist__right_knee__left_foot'] < 0.03:
        if features['angle__right_thigh__right_calf'] < 170:
            inaccurates.append('Right leg is not straight')
        elif features['angle__right_upper__right_lower'] < 160:
            inaccurates.append('Body is not straight')
            
    else:
        inaccurates.append('Not bending leg at knee')
    
    return inaccurates