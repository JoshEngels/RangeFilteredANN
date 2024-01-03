#include "neighbor.h"

namespace diskann {

  // Wrapper insert method that determines whether to double the beam and add best nonvisited nodes
  

  void NeighborPriorityQueue::insert(const Neighbor &nbr)  {
        if (_try_new_beamsearch) {
          _extra_nodes.push(nbr);
        }

        if (_size == _capacity && _data[_size - 1] < nbr)
        {
            return;
        }

        size_t lo = 0, hi = _size;
        while (lo < hi)
        {
            size_t mid = (lo + hi) >> 1;
            if (nbr < _data[mid])
            {
                hi = mid;
                // Make sure the same id isn't inserted into the set
            }
            else if (_data[mid].id == nbr.id)
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
        _data[lo] = {nbr.id, nbr.distance};
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
        return _cur < _size;
    }
    
    } // namespace diskann