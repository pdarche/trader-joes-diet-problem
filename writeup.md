# The Trader Joe's Diet Problem

"Adulting," as we Millenials say, is hard. It requires doing so many difficult things, like managing finances, finding a partner, establishing a career, and, most basic &ndash; yet often most difficult &ndash; of all, feeding yourself. You see, we want food that simulaneously cheap enough to match our debt-laden finances, and is nutritous enough to match whatever dietary restrictions we happen to be abiding by at the moment. This is no mean feat! Luckily, our fore-parents already thought about this problem and both have a name for it and a mathematical framework for solving it.

#### The problem
That name is the [_diet problem_](https://neos-guide.org/content/diet-problem) and that framework is [linear programming](https://en.wikipedia.org/wiki/Linear_programming). The problem is specified as follows: "given a set of foods, along with the nutrient information for each food and the cost per serving of each food, select the number of servings of each food to purchase (and consume) so as to minimize the cost of the food while meeting the specified nutritional requirements." (description borrowed from linked page).

That is, give a table of food items like this:

| name   |   cost |   fat |   protein |   carbs |   ... | calories |
|:-------|-------:|------:|----------:|--------:|------:|---------:|
| Food 1 |   1.00 |  100  |        16 |     126 |   ... |   132    |
| Food 2 |   0.6  |  83   |         8 |     80  |   ... |   318    |
| Food 3 |   0.75 |   41  |        20 |     123 |   ... |   151    |
| ... |   |    |         |      |   ... |     |
| Food n |   0.75 |   41  |        20 |     123 |   ... |   151    |

we want to pick the right number of servings of each food to purchase: $`amount_{1}, amount_{2}, ... , amount_{n}`$

so that the total cost: $`amount_{1} * cost_{1} + amount_{2} * cost_{2} + ... + amount_{n} * cost_{n}`$ is minimized

and all our nutritional constraints:

```math
amount_{1} * calories_{1} + amount_{2} * calories_{2} + ... + amount_{n} * cost_{n} \leq 2400 \\ 
amount_{1} * fat_{1} + amount_{2} * fat_{2} + ... + amount_{n} * fat_{n} \leq 70 \\
amount_{1} * carbs_{1} + amount_{2} * carbs_{2} + ... + amount_{n} * carbs_{n} \leq 300 \\
amount_{1} * protein_{1} + amount_{2} * protein_{2} + ... + amount_{n} * protein_{n} \leq 300 \\

```

are satisfied.

Writing out all of these linear combinations is tedious, so we express the problem using vectors and matricies thusly:

```math
minimize \ c^{T}x \\
subject \ to \ Ax \leq b \\
and \ x \geq 0
```

Here $`c`$ is a vector with $`n`$ components, each representing the cost per serving of a food item. $`x`$ a vector with $`n`$ components representing the number of servings of each food item we'll purchase. $`A`$ is our matrix of nutritional information. Each of the $`n`$ rows represents a food item, and each of the $`m`$ columns giving different nutritional information for one serving of the food.  $`b`$ is a vector with $`k`$ components, each representing a the value of a constraint that must be satisfied (i.e. number of calories).  The first line in the problem is our _objective_ and the second two are our _constraints_. The final constrait $`x \geq 0`$ represents the fact that we should never have less than 0 servings of a given food.

That's the general form of the problem. In our case, we want to minimize cost while sticking to different kinds of diets. To keep things simple, we'll use diets that are defined by the proportion of calories that come from protein, fat, and carbohydrates, or that have some sort of nutritional restriction (like being vegan) that we can filter down to. Those diets are:

| diet          |   % fat| % protein | % carbohydrates |
|:--------------|-------:|----------:|----------------:|
| Balanced      | 30     | 30        | 40              |
| Keto          | 75     | 24        | 1               |
| Atkins        | 45     | 45        | 10              |
| Vegan         | 30     | 30        | 45              |
| Gluten-free   | 30     | 30        | 45              |

(For the vegan and gluten-free diets, we use the "balanced" macronutrient proportions.) For simplicity, we also add a constraint that the number of servings has to be an integer and restrict the total number of servings to four per food item so we make sure we get a little variety. Lastly, we'll use a recommended calorie count for roughly one day's worth of food.

### The data
Unfortunately, grocers often don't make available an easily accessible table of foods with cost and nutrtition information. I mostly buy groceries at Trader Joe's and they definitely don't make this information avaiable (I asked). Fortunately, they _do_ publish their [Fealess Flyer](https://www.traderjoes.com/fearless-flyer) online, complete with individual articles for various food items that include price and nutrition information. And _especially_ fortunately, they use a url scheme for those articles that identifies products by their index, which happens to just be an integer. This means we can iterate over the pages and generate exactly the data we're looking for.

We end up with 329 products with complete information (out of 583 total), including price, number of servings, calories, protein, fat, and carbohydrates (the dataset contains 17 nutritional features, but only four are used). The macronutrient columns for the final dataset were transformed to be in units of calories rather than grams to simplify the expression of the constraints. The problem data looks like this:


| name                           |   cost_per_serving |         fat |   protein |         carbs |            calories |
|:-------------------------------|-------------------:|------------:|----------:|--------------:|--------------------:|
| For the Love of Chocolate Cake |            1.33    |       180   |        16 |           176 |               372   |
| Pancake Bread                  |            0.49875 |        90   |         8 |           100 |               198   |
| Pizza Crusts                   |            0.43625 |        22.5 |        20 |           112 |               154.5 |
| Cold Pressed Green Juice       |            3.99    |         0   |        20 |            80 |               100   |
| French Fromage Slices          |            0.62375 |        81   |        20 |             0 |               101   |
| ...                            |                    |             |           |               |                     |

The full dataset is [here](https://github.com/pdarche/trader-optimal/blob/master/data/clean_flyer_data.csv)

### The code
With data in hand, we can solve for $`x`$. To do so we write the following small [CVXPY](https://github.com/cvxgrp/cvxpy/) program:

```python
  # Packages
  import cvxpy as cp
  import pandas as pd

  # Problem data
  df = pd.read_csv('./data/clean/fearless_flyer.csv')

  # Variables
  n = len(df)
  c = df['cost_per_serving'].values
  A = df[['calories', 'fat', 'protein', 'carbs']].values

  # Constraints for a "balanced" diet
  min_cals = 2200
  max_cals = 2800
  fat = .3
  protein = .3
  carbs = .4

  constraints = [
      A[:, 0]*x >= min_cals,
      A[:, 0]*x <= max_cals,
      A[:, 1]*x >= min_cals * fat,
      A[:, 1]*x <= max_cals * fat,
      A[:, 2]*x >= min_cals * protein,
      A[:, 2]*x <= max_cals * protein,
      A[:, 3]*x >= min_cals * carbs,
      A[:, 3]*x <= max_cals * carbs,
      x >= 0,
      x <= max_servings
  ]  

  # Optimization variable and objective
  x = cp.Variable(n, integer=True)
  objective = cp.Minimize(c.T*x)
  
  # Problem
  problem = cp.Problem(objective, constraints)
  problem.solve()

```

That's it! In CVXPY, linear programs are programmed very similarly to how they're expressed mathematically. Above we create the variables mentioned earlier: `c`, the vector of costs for each food, `x`, the CVXPY `Variable` to be optimized, `A` the matrix of foods with their nutrition information. To create our objective we pass the function to be minimized into `Minimize`. The constraints above aren't exactly expressed as the linear inequality $`Ax \leq b`$, but they could be. Once we have our objective and constraints we create an instance of the `Problem` class, and then call `solve`.


#### The solution
So what do we get? We access the optimized values through the `value` attribute of our `x` optimization variable after a feasible solution has been found, and multiply that by the cost per serving and nutritional information to get the final results for each diet: 


__Balanced__

| name                                              |   total_fat |   protein |   total_carbs |   calories |   servings |   cost_per_serving |     cost |
|:--------------------------------------------------|------------:|----------:|--------------:|-----------:|-------------------:|-------------------:|---------:|
| [Sprouted Wheat Sourdough](https://www.traderjoes.com/fearless-flyer/article/4655)                          |           0 |       112 |           224 |        336 |                  4 |           0.1995   | 0.798    |
| [Organic High Protein Tofu](https://www.traderjoes.com/fearless-flyer/article/4698)                         |         252 |       224 |            48 |        524 |                  4 |           0.498    | 1.992    |
| [Creamy Salted Peanut Butter](https://www.traderjoes.com/fearless-flyer/article/4930)                       |         432 |        84 |            84 |        600 |                  3 |           0.142143 | 0.426429 |
| [Pumpkin Pancake Mixes, Gluten-Full or Gluten-Free](https://www.traderjoes.com/fearless-flyer/article/5048) |           0 |        48 |           432 |        480 |                  3 |           0.165833 | 0.4975   |
| [Mini Vegetable Samosas](https://www.traderjoes.com/fearless-flyer/article/4800)                            |         126 |       192 |           160 |        478 |                  2 |           1.16333  | 2.32667  |
| __total__                                         |     __810__ |   __660__ |       __948__ |   __2418__ |             __16__ |                    |__6.06__ |


__Keto__

| name                          |   total_fat |   protein |   total_carbs |   calories |   servings |   cost_per_serving |     cost |
|:------------------------------|------------:|----------:|--------------:|-----------:|-------------------:|-------------------:|---------:|
| [Organic EVOO from Spain](https://www.traderjoes.com/fearless-flyer/article/4769)       |         504 |         0 |             0 |        504 |                  4 |           0.166364 | 0.665455 |
| [Unexpected Cheddar](https://www.traderjoes.com/fearless-flyer/article/4787)            |         360 |       112 |             0 |        472 |                  4 |           0.57     | 2.28     |
| [Cornish Game Hens](https://www.traderjoes.com/fearless-flyer/article/5185)             |         576 |       304 |             0 |        880 |                  4 |           0.886667 | 3.54667  |
| [Organic High Protein Tofu](https://www.traderjoes.com/fearless-flyer/article/4698)     |         126 |       112 |            24 |        262 |                  2 |           0.498    | 0.996    |
| [Black Truffle Butter](https://www.traderjoes.com/fearless-flyer/article/5184)          |          99 |         0 |             0 |         99 |                  1 |           0.498333 | 0.498333 |
| __Total__                     |   __1665__  |   __528__ |        __24__ |   __2217__ |             __15__ |                    | __7.99__ |


__Atkins__

| name                              |   total_fat |   protein |   total_carbs |   calories |   servings |   cost_per_serving |    cost |
|:----------------------------------|------------:|----------:|--------------:|-----------:|-------------------:|-------------------:|--------:|
| [Organic High Protein Tofu](https://www.traderjoes.com/fearless-flyer/article/4698)         |         252 |       224 |            48 |        524 |                  4 |           0.498    | 1.992   |
| [All Natural Ground Turkey Patties](https://www.traderjoes.com/fearless-flyer/article/4892) |         288 |       320 |             0 |        608 |                  4 |           1.1225   | 4.49    |
| [Cornish Game Hens](https://www.traderjoes.com/fearless-flyer/article/5185)                 |         576 |       304 |             0 |        880 |                  4 |           0.886667 | 3.54667 |
| [Sprouted Wheat Sourdough](https://www.traderjoes.com/fearless-flyer/article/4655)          |           0 |        56 |           112 |        168 |                  2 |           0.1995   | 0.399   |
| [Mini Vegetable Samosas](https://www.traderjoes.com/fearless-flyer/article/4800)            |          63 |        96 |            80 |        239 |                  1 |           1.16333  | 1.16333 |
| __Total__                         |    __1179__ |  __1000__ |       __240__ | __2419__ |               __15__ |                    |__11.59__|


__Vegan__

| name                            |   total_fat |   protein |   total_carbs |   calories |   servings |   cost_per_serving |     cost |
|:--------------------------------|------------:|----------:|--------------:|-----------:|-------------------:|-------------------:|---------:|
| [Organic High Protein Tofu](https://www.traderjoes.com/fearless-flyer/article/4698)       |       252   |       224 |            48 |      524   |                  4 |           0.498    | 1.992    |
| [Japanese Style Fried Rice](https://www.traderjoes.com/fearless-flyer/article/4741)       |       180   |       112 |           672 |      964   |                  4 |           0.854286 | 3.41714  |
| [Vegan Banana Bread with Walnuts](https://www.traderjoes.com/fearless-flyer/article/5021) |       243   |        24 |           360 |      627   |                  3 |           0.49875  | 1.49625  |
| [Soft-Baked Snickerdoodles](https://www.traderjoes.com/fearless-flyer/article/4945)       |        40.5 |         4 |            72 |      116.5 |                  1 |           0.498333 | 0.498333 |
| __total__                       |     __715__ |   __364__ |      __1152__ | __2231.5__ |             __12__ |                    |__7.40__|


__Gluten Free__

| name                                    |   total_fat |   protein |   total_carbs |   calories |   servings |   cost_per_serving |     cost |
|:----------------------------------------|------------:|----------:|--------------:|-----------:|-------------------:|-------------------:|---------:|
| [Pizza Crusts](https://www.traderjoes.com/fearless-flyer/article/5122)                            |          90 |        80 |           448 |        618 |                  4 |           0.43625  | 1.745    |
| [Organic Creamy Tomato Soup](https://www.traderjoes.com/fearless-flyer/article/4691)              |          72 |        80 |           256 |        408 |                  4 |           0.6725   | 2.69     |
| [Neapolitan Puffs Cereal](https://www.traderjoes.com/fearless-flyer/article/4805)                 |          54 |        96 |           432 |        582 |                  4 |           0.527143 | 2.10857  |
| [Organic Grass-Fed Uncured Beef Hot Dogs](https://www.traderjoes.com/fearless-flyer/article/4896) |         360 |       144 |            16 |        520 |                  4 |           1.198    | 4.792    |
| [Spicy Cheese Crunchies](https://www.traderjoes.com/fearless-flyer/article/4909)                  |         108 |        16 |           136 |        260 |                  2 |           0.284286 | 0.568571 |
| [Organic Blue Corn Tortilla Chips](https://www.traderjoes.com/fearless-flyer/article/4711)        |          81 |        12 |            68 |        161 |                  1 |           0.249167 | 0.249167 |
| [Organic Hemp Seed Bars](https://www.traderjoes.com/fearless-flyer/article/4845)                  |          63 |        12 |            44 |        119 |                  1 |           0.598    | 0.598    |
|                               __total__ |     __828__ |   __440__ |      __1400__ |  __2668__  |                 20 |                    |__12.75__ |

### Discussion
These results are ... not that bad! I mean, the vegan food combination sounds delicious. And, though I don't eat much meat, if I did, I'd defintely be happy with Cornish game hen with truffle butter and olive oil. It's interesting to see what's being picked up on. The program was clearly finding the items with the highest amounts of protein and fat per dollar. For protein, those items were: 1) creamy salted peanut butter, 2) sprouted wheat sourdough, 3) high protein tofu, 4) pumpkin pancake mix, and 4) the game hens, and most of those items were used when possible.

That said, I don't think I'll be making long-term dietary plans around this though. I could probably make some improvements by adding in constraints around vitamins or other nutritional features, but this approach is pretty limited and leaves out important parts of what goes into choosing a meal (like how much you enjoy a given food, how well each foods go together, how long it takes to prepare them, etc). But, all things considered, the output is pretty good. And even better, the dataset has a lot of valuable information to use in other analyses. So even if this doesn't actually solve the problem in any realistic way, it was worth trying.

