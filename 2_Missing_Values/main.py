# Sometimes, datasets that include empty data cells have to be used. There are some ways to cope with this kind of issues. In this chapter, we will see the most
# common ones.
# 1-Drop columns with missing values: it's not the most suitable one, as a whole feature gets removed from the dataset. For example, while analyzing real estate
# data, dropping a column may imply removing data about the usable surface extension in a home/flat.
# 2-Imputation: instead of removing any value, some values may be introduced arbitrarily within the dataset. And yes, the data can be made up, but the dataset
# (and models that use the dataset in question) usually behave better than those which just remove the whole feature from the dataset. The inputted value can
# be something like a mean value of the other values found in the same dataset column.
# 3-Imputation with extra data: sometimes, which values were originally missing has to be taken into account. To do so, another column may be added next to the
# one that has missing values within it, telling if that row within that column included data originally or not.