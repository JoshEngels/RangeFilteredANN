#include "neighbor.h"

namespace diskann {
  

  void NeighborPriorityQueue::insert(const Neighbor &nbr, float filter_value)  {
        if (_try_new_beamsearch) {
          _extra_nodes.push(nbr);
          bool meets_constraint = filter_value >= _range_filter->first && filter_value <= _range_filter->second;
          _meets_constraint[nbr.id] = meets_constraint;
        } else {
          _meets_constraint[nbr.id] = true;
        }


        if (_size == _capacity && _data[_size - 1] < nbr)
        {
            if (_try_new_beamsearch)
            {
              auto num_locked = num_locked_in_filtered();
              if (num_locked >= _goal_num_filtered) {
                // std::cout << "YES " << num_locked << " " << _capacity << std::endl;
                return;
              } else {
                // std::cout << "NO " << num_locked << " " << _capacity << std::endl;
                reserve(_capacity * 2);
              }
            }
            else {
              return;
            }
        }

        auto current_neighbor = _try_new_beamsearch? _extra_nodes.top() : nbr;

        // if (_try_new_beamsearch) {
        //   std::cout << "NEIGHBOR " << current_neighbor.id << " " << _cur << " " << _try_new_beamsearch << " " << _capacity << std::endl;
        // }
        if (_try_new_beamsearch) {
          _extra_nodes.pop();
        }

        size_t lo = 0, hi = _size;
        while (lo < hi)
        {
            size_t mid = (lo + hi) >> 1;
            if (current_neighbor < _data[mid])
            {
                hi = mid;
                // Make sure the same id isn't inserted into the set
            }
            else if (_data[mid].id == current_neighbor.id)
            {
                return;
            }
            else
            {
                lo = mid + 1;
            }
        }

        if (lo < _capacity)
        {
            std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Neighbor));
        }
        _data[lo] = {current_neighbor.id, current_neighbor.distance};
        if (_size < _capacity)
        {
            _size++;
        }
        if (lo < _cur)
        {
            _cur = lo;
        }
    }

    bool NeighborPriorityQueue::has_unexpanded_node() const
    {
        if (_try_new_beamsearch)
        {
            return num_locked_in_filtered() < _goal_num_filtered;
        }
        else
        {
            return _cur < _size;
        }
    }

    size_t NeighborPriorityQueue::num_locked_in_filtered () const
    {
        size_t num_locked = 0;
        for (size_t i = 0; i < _size && _data[i].expanded; i++)
        {
            if (_meets_constraint.at(_data[i].id))
            {
                num_locked++;
            }
        }
        return num_locked;
    }
    } // namespace diskann