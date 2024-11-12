# Disclinations
Plates with...

.

.
.
.
.
.
.
.
.
.
.
.
.
.
.
.

### Prerequisites

The project assumes basic knowledge of the theory of infinitesimal elasticity and
finite element methods.

Basic knowledge of Python will be assumed, see https://github.com/jakevdp/WhirlwindTourOfPython
to brush up if you feel unsure.

Basic knowledge of git as a versioning system with feature-branch workflow
https://gist.github.com/brandon1024/14b5f9fcfd982658d01811ee3045ff1e

Remember to set your name and email before pushing to the repository,
either locally or globally, see https://www.phpspiderblog.com/how-to-configure-username-and-email-in-git/

#### Feature branch workflow

For each new feature you wish to implement, create a branch named ```{yourname}-{feature}```, 
as in ```andres-meshes```.

https://docs.gitlab.com/ee/gitlab-basics/feature_branch_workflow.html

 - Create your feature branch:`git checkout -b username-feature`
 - To push your branch: `git push -u origin feature_branch_name`
 - Create a pull request on the main branch for merging. Somebody should approve the pull-request. -

### Weekly updates (merge from main)
```
git checkout main
git pull
git checkout yourname-branch
git merge main
```

Asymmetrically, feature-work is `rebased`.

### To run the code (on Docker)

First, run the container, attaching an interactive session and sharing data space 
(the current dir) between the host and the container (the syntax is origin:target).

On a Mac:
```
docker run --rm -ti -v "$(pwd)":/home/numerix -w /home/numerix kumiori3/numerix:latest
```

On a Windows:
```
docker run --rm -ti -v "C:/...":/home/numerix" -w /home/numerix kumiori3\numerix:latest
```

### To run the tests, execute the following command in your terminal:
```
pytest tests
```

### Authors
- Cf. commit messages
  
