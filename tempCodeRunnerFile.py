
        world_landmarks = result.pose_world_landmarks[0]
        rightWrist = world_landmarks[16]
        l.append((rightWrist.x, rightWrist.y, rightWrist.z))