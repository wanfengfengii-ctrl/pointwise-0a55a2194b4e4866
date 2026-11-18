from review.projection import ResidueClass, project_channel


def evaluate_joint_claim(relation, channels):
    for channel in channels:
        reachable = project_channel(relation, channel.coefficients)
        if isinstance(reachable, ResidueClass):
            if channel.claimed % reachable.step != reachable.residue:
                return False
        elif channel.claimed != reachable:
            return False
    return True
