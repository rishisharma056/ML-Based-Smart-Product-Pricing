# ML Challenge 2025 Problem Statement Template 

## About the Hackathon :

Amazon ML Challenge is a two-stage competition where students from all engineering campuses across India will get a unique opportunity to work on Amazon’s dataset to bring in fresh ideas and build innovative solutions for a real-world problem statement. The top three winning teams will receive cash prizes and certificates.

### Eligibility and Team Rules:

- This competition is open to all students pursuing PhD/ M.E./M.Tech./ M.S./MS by Research/B.E./B.Tech. full-time degree across all          engineering campuses in India. 
- Graduation Year: 2026 or 2027
- Each team must consist of a minimum of 3 and a maximum of 4 members.
- Each team must have a team leader.
- Cross-college teams are allowed.
- One student cannot be a member of more than one team.

### Competition Structure:
#### Round 1 - ML Hackathon:
 - All teams will get access to the problem statement with the data set on day 1 and will have time to build and submit solutions till day 3.
 - Teams can track their performance through the leaderboard, which will reflect team rankings live over the course of this challenge.
 - Each team should upload a document between 1 and 2 pages detailing their approach to the solution, along with a zipped file containing the code, script, or notebook.

#### Round 2 - Finale : 
Based on the leaderboard results and the solution presented in the document at the end of Round 1, the top 10 teams will be invited to the Grand Finale to present their solutions to Amazon's Scientists

## About this Repository : 

This repository has two branches. 

1. **Problem-Statement-Template** This is the initial file structure. You can find the problem statement at (ProblemStatement.md)
2. **Completed** This branch contains the solution we worked on. 

## Want to work on this problem statement?

Fork the **Problem-Statement-Template** branch. 
```bash
git checkout Problem-Statement-Template
```

## Want to view my solution ?

Change the branch to **Completed**
```bash
git checkout Completed
```

## File Descriptions:

*Source files*

1. **src/utils.py:** Contains helper functions for downloading images from the image_link. You may need to retry a few times to download all images due to possible throttling issues.
2. **sample_code.py:** Sample dummy code that can generate an output file in the given format. Usage of this file is optional.

*Dataset files*

1. **dataset/train.csv:** Training file with labels (`price`).
2. **dataset/test.csv:** Test file without output labels (`price`). Generate predictions using your model/solution on this file's data and format the output file to match sample_test_out.csv
3. **dataset/sample_test.csv:** Sample test input file.
4. **dataset/sample_test_out.csv:** Sample outputs for sample_test.csv. The output for test.csv must be formatted in the exact same way. Note: The predictions in the file might not be correct

### Note

Due to file size constraints, the training and test datasets are not included in this repository.


