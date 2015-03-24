#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#define DEBUG_MODE 0

using namespace std;


template<typename TimeT = std::chrono::milliseconds>
struct measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep execution(F func, Args&&... args)
    {
        auto start = std::chrono::system_clock::now();
        func(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast< TimeT>
                            (std::chrono::system_clock::now() - start);
        return duration.count();
    }
};

class Interval {
 public:
  int left, right;

  Interval(int left, int right) : left(left), right(right) {}

  bool Contains(int x) const {
    return left <= x && x <= right;
  }

  // Returns true if the interval is to the left of the point.
  bool ToTheLeft(int x) const {
    return x > right;
  }

  // Returns true if the interval is to the right of the point.
  bool ToTheRight(int x) const {
    return x < left;
  }

  void Print() const {
    cout << "[" << left << ", " << right << "]";
  }
};

// Decides left or right by comparing the data at dimension_index versus the
// ref_value.
class Decision {
 public:
  Decision() : dimension_index(-1), ref_value(-1) {}
  Decision(int dimension_index, int ref_value)
      : dimension_index(dimension_index), ref_value(ref_value) {}

  bool IsValid() const {
    assert(dimension_index >= -1);
    return dimension_index >= 0;
  }

  void Print() const {
    cout << "Decision(dim = " << dimension_index << ", ref = " << ref_value << ")";
  }

  int dimension_index, ref_value;
};

class DecisionTreeNode {
 public:
  DecisionTreeNode() : left(nullptr), right(nullptr) {
  }

  bool FindPoint(const vector<int>& point) {
    // First look into the current node.
    if (CurrentNodeContainsPoint(point)) {
      return true;
    }

    // If no decision to make in this node, then we couldn't find it.
    if (!decision.IsValid()) {
      assert(left == nullptr && right == nullptr);
      return false;
    }

    if (point[decision.dimension_index] < decision.ref_value) {
      if (left != nullptr) {
        return left->FindPoint(point);
      }
    } else if (point[decision.dimension_index] > decision.ref_value) {
      if (right != nullptr) {
        return right->FindPoint(point);
      }
    }
    return false;
  }

  void Print(string indent = "") const {
    cout << indent << "NODE" << endl;

    cout << indent;
    decision.Print();
    cout << endl;

    if (left != nullptr) {
      cout << indent << "Left child:" << endl;
      left->Print(indent + "  ");
    } else {
      cout << indent << "Left child missing." << endl;
    }

    if (right != nullptr) {
      cout << indent << "Right child:" << endl;
      right->Print(indent + "  ");
    } else {
      cout << indent << "Right child missing." << endl;
    }
  }

  void FreeBeneath() {
    if (left != nullptr) {
      left->FreeBeneath();
      delete left;
    }
    if (right != nullptr) {
      right->FreeBeneath();
      delete right;
    }
  }

  size_t SumAreasToCheckSizes() {
    size_t sz = areas_to_check.size();
    if (left != nullptr) {
      sz += left->SumAreasToCheckSizes();
    }
    if (right != nullptr) {
      sz += right->SumAreasToCheckSizes();
    }
    return sz;
  }

  // Exposed data.
  vector<vector<Interval>> areas_to_check;
  DecisionTreeNode* left;
  DecisionTreeNode* right;
  Decision decision;

 private:

  bool CurrentNodeContainsPoint(const vector<int>& point) {
    for (const vector<Interval>& area : areas_to_check) {
      bool current_area_contains_point = true;
      for (int i = 0; i < (int) point.size(); ++i) {
        if (!area[i].Contains(point[i])) {
          current_area_contains_point = false;
          break;
        }
      }
      if (current_area_contains_point) {
        return true;
      }
    }
    return false;
  }
};

class DecisionTree {
 public:
  DecisionTree(int dimension)
      : dimension_(dimension), dirty_(true), root_(nullptr) {
  }

  ~DecisionTree() {
    FreeTree();
  }

  void AddArea(const vector<Interval>& area) {
    assert(static_cast<int>(area.size()) == dimension_);
    areas_.push_back(area);
    dirty_ = true;
  }

  bool ContainsPoint(const vector<int>& point) {
    assert(static_cast<int>(point.size()) == dimension_);
    RebuildTreeIfDirty();
    return root_->FindPoint(point);
  }

  void RebuildTreeIfDirty() {
    if (dirty_) {
      RebuildTree();
      dirty_ = false;
    }
  }

  void RebuildTree() {
    FreeTree();
    root_ = BuildTree_(areas_);
  }

  void Print() {
    if (root_ == nullptr) {
      cout << "Root node is null." << endl;
    } else {
      cout << "Decision tree:" << endl;
      root_->Print();
    }
  }

  size_t SumAreasToCheckSizes() {
    return root_->SumAreasToCheckSizes();
  }

  const vector<vector<Interval>>& GetAllAreas() const {
    return areas_;
  }

 private:
  int dimension_;
  bool dirty_;
  DecisionTreeNode* root_;
  vector<vector<Interval>> areas_;

  void FreeTree() {
    if (root_ != nullptr) {
      root_->FreeBeneath();
      delete root_;
    }
  }

  DecisionTreeNode* BuildTree_(const vector<vector<Interval>>& areas_to_add) {
    assert(areas_to_add.size() > 0);

    DecisionTreeNode* node = new DecisionTreeNode();

    // If we only have one area, then check it in the node and that's it.
    if (areas_to_add.size() == 1) {
      node->areas_to_check = areas_to_add;
      return node;
    }

    Decision best_decision(-1, -1);
    int best_cut_through_count = -1;
    vector<vector<Interval>> all_areas(areas_to_add);
    for (int dimension_index = 0; dimension_index < dimension_; ++dimension_index) {
      // Sort the areas by the right side of the interval at dimension_index so
      // we can easily tell the middle of the vector.
      sort(all_areas.begin(), all_areas.end(),
          [dimension_index] (const vector<Interval>& a,
                             const vector<Interval>& b) {
            return a[dimension_index].right < b[dimension_index].right;
          });

      // We know we have at least two areas to distinguish between, we look at the median element.
      int median_index = ((int) all_areas.size() - 1) / 2;

      // Then we cut the space the following way: everything that is on the left
      // will be pushed to the left node. Everighing that is strictly higher than
      // the middle will be pushed to the right node. Everything that is cut
      // through will be in the current node to check.
      Decision decision(dimension_index,
                        all_areas[median_index][dimension_index].right + 1);
#if DEBUG_MODE
      cout << "Examining ";
      decision.Print();
      cout << endl;
#endif

      // The best decision is the one that cuts through as few areas as possible.
      int cut_through_count = 0, left_count = 0, right_count = 0;
      for (const vector<Interval>& area : all_areas) {
        if (area[dimension_index].Contains(decision.ref_value)) {
          ++cut_through_count;
        } else if (area[dimension_index].ToTheLeft(decision.ref_value)) {
          ++left_count;
        } else if (area[dimension_index].ToTheRight(decision.ref_value)) {
          ++right_count;
        } else {
          assert(false);
        }
      }

      if (best_decision.dimension_index == -1 ||
          cut_through_count < best_cut_through_count) {
        best_decision = decision;
        best_cut_through_count = cut_through_count;
      }
    }

#if DEBUG_MODE
    cout << "Best decision: ";
    best_decision.Print();
    cout << endl;
    cout << "Best cut through: " << best_cut_through_count << endl;
#endif

    // Check that we actually made a choice.
    assert(best_decision.dimension_index != -1);

    vector<vector<Interval>> push_left_areas, push_right_areas;

    for (const vector<Interval>& area : areas_to_add) {
      const Interval& interval_for_decision = area[best_decision.dimension_index];
      if (interval_for_decision.Contains(best_decision.ref_value)) {
        node->areas_to_check.push_back(area);
      } else if (interval_for_decision.ToTheLeft(best_decision.ref_value)) {
        push_left_areas.push_back(area);
      } else if (interval_for_decision.ToTheRight(best_decision.ref_value)) {
        push_right_areas.push_back(area);
      } else {
        assert(false);
      }
    }

#if DEBUG_MODE
    cout << "pushing left: " << push_left_areas.size() << endl;
    cout << "pushing right: " << push_right_areas.size() << endl;
#endif

    // If all data goes left or right we will end up with infinite recursion
    // (the same) will happen at the next call, so better keep everything in
    // the current node.
    if (push_left_areas.size() == areas_to_add.size() ||
        push_right_areas.size() == areas_to_add.size()) {
      node->areas_to_check = areas_to_add;
#if DEBUG_MODE
      cout << "Avoiding infinite recursion." << endl;
#endif
      return node;
    }

    // Finally, add the decision to the current node and build left and right
    // children.
    node->decision = best_decision;

    if (!push_left_areas.empty()) {
      node->left = BuildTree_(push_left_areas);
    }

    if (!push_right_areas.empty()) {
      node->right = BuildTree_(push_right_areas);
    }

    return node;
  }
};

void SimpleTest() {
  DecisionTree* decision_tree = new DecisionTree(2);

  // Just for testing, we add a few squares in 2D and then look for points
  // inside and outside them.
  decision_tree->AddArea({ Interval(1, 5), Interval(1, 5) });
  decision_tree->AddArea({ Interval(1, 5), Interval(11, 15) });
  decision_tree->AddArea({ Interval(11, 15), Interval(11, 15) });
  decision_tree->AddArea({ Interval(11, 15), Interval(1, 5) });

  decision_tree->RebuildTree();

  assert(decision_tree->ContainsPoint({3, 3}) == true);
  assert(decision_tree->ContainsPoint({20, 20}) == false);

  delete decision_tree;

  cout << "SimpleTest passed." << endl;
}

bool BruteContains(const vector<int>& point,
                   const vector<vector<Interval>>& areas) {
  for (const vector<Interval>& area : areas) {
    bool current_area_contains_point = true;
    for (int i = 0; i < (int) point.size(); ++i) {
      if (!area[i].Contains(point[i])) {
        current_area_contains_point = false;
        break;
      }
    }
    if (current_area_contains_point) {
      return true;
    }
  }
  return false;
}

template<int D>
void RandomTest(int add_count,
                int contains_count,
                bool check_for_correctness = false) {
  cout << "RandomTest (D = " << D << ") begins." << endl;
  DecisionTree* dt = new DecisionTree(D);

  int offset_mod = 100000;
  cout << "add areas speed for " << add_count << " instances: "
       << measure<>::execution([&] {
            for (int i = 0; i < add_count; ++i) {
              vector<Interval> area;
              for (int d = 0; d < D; ++d) {
                int offset = rand() % offset_mod;
                area.push_back(Interval(offset + rand() % 100, offset + 100 + rand() % 100));
              }
              dt->AddArea(area);
            }
          }) << "ms" << endl;

  cout << "build tree speed for " << add_count << " instances: "
       << measure<>::execution([&] { dt->RebuildTree(); }) << "ms" << endl;

  cout << "contains speed for " << contains_count << " instances: "
       << measure<>::execution([&] {
           for (int i = 0; i < contains_count; ++i) {
             vector<int> point;
             for (int d = 0; d < D; ++d) {
               point.push_back(rand() % offset_mod);
             }
             bool result = dt->ContainsPoint(point);
             if (check_for_correctness) {
               assert(result == BruteContains(point, dt->GetAllAreas()));
             }
           }
         }) << "ms" << endl;

  assert(dt->SumAreasToCheckSizes() == add_count);
  delete dt;

  cout << "RandomTest (D = " << D << ") passed." << endl;
}

int main(int argc, char* argv[]) {
  // tiny manual correctness test.
  SimpleTest();
  // 2D stress test.
  RandomTest<2>(1000, 1000000);
  // ipv4 like test (4 groups).
  RandomTest<4>(1000, 1000000);
  // ipv6 like test (8 groups).
  RandomTest<8>(1000, 1000000);
  // correctness stress test.
  RandomTest<2>(1000, 100000, true);

  return 0;
}
