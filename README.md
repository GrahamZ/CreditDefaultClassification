# CreditDefaultClassification

In these notebooks, we analyse a data set containing bank loan records, labelled as either a good loan that was repaid or a bad loan where the customer defaulted. The data set was obtained from the Open ML web site at https://www.openml.org/d/31, but originally came from the UCI repository (Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science).

The data set contains 1000 labelled examples with 20 features, of which 700 are good loans and 300 are bad loans. So the data set is imbalanced but not extremely so. It is conventional to assign the minority class to be the positive class (bad loans).

The task is to learn a binary classifier that will predict if a loan to an applicant will be good or bad. To make this task well defined, we will set a target metric that the classifier should achieve.

First, the bank would prefer to avoid making bad loans, so the priority is to identify a high proportion of the positive class (bad loans). Therefore, we set a target of a recall metric of at least 80%.

Second, the data set comes with a cost matrix that specifies the cost of mis-classifying a loan:
"It is worse to class a customer as good when they are bad (5), than it is to class a customer as bad when they are good (1)."
We will use these to obtain a classifier that minimises the cost.

The fundamentals of cost sensitive learning are described in the following paper:
Elkan, C.: 2001, ‘The Foundations of Cost-Sensitive Learning’. In: Proceedings of the IJCAI-01. pp. 973–978.
