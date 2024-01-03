// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <mutex>
#include <queue>
#include <vector>
#include "utils.h"

inline std::string getEnvVar( std::string const & key )
{
    char * val = getenv( key.c_str() );
    return val == NULL ? std::string("") : std::string(val);
}

namespace diskann
{

struct Neighbor
{
    unsigned id;
    float distance;
    bool expanded;

    Neighbor() = default;

    Neighbor(unsigned id, float distance) : id{id}, distance{distance}, expanded(false)
    {
    }

    inline bool operator<(const Neighbor &other) const
    {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }

    inline bool operator==(const Neighbor &other) const
    {
        return (id == other.id);
    }
};

// Invariant: after every `insert` and `closest_unexpanded()`, `_cur` points to
//            the first Neighbor which is unexpanded.
class NeighborPriorityQueue
{
  public:
    NeighborPriorityQueue() : _size(0), _capacity(0), _cur(0), _goal_num_filtered(0)
    {
    }

    explicit NeighborPriorityQueue(size_t capacity) : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1), _goal_num_filtered(capacity)
    {
    }

    // Inserts the item ordered into the set up to the sets capacity.
    // The item will be dropped if it is the same id as an exiting
    // set item or it has a greated distance than the final
    // item in the set. The set cursor that is used to pop() the
    // next item will be set to the lowest index of an uncheck item
    void insert(const Neighbor &nbr, float filter_value = 0);

    Neighbor closest_unexpanded()
    {
        _data[_cur].expanded = true;
        size_t pre = _cur;
        while (_cur < _size && _data[_cur].expanded)
        {
            _cur++;
        }
        return _data[pre];
    }

    bool has_unexpanded_node() const;

    size_t size() const
    {
        return _size;
    }

    size_t capacity() const
    {
        return _capacity;
    }

    void reserve(size_t capacity)
    {
        if (capacity + 1 > _data.size())
        {
            _data.resize(capacity + 1);
        }
        _capacity = capacity;
    }

    void set_filter(const std::pair<float, float> &range_filter, size_t goal_num_filtered)
    {
        if (range_filter.second > range_filter.first) {
          _range_filter = range_filter;
          _goal_num_filtered = goal_num_filtered;
          _try_new_beamsearch = true;
        } else {
          _try_new_beamsearch = false;
        }
    }

    Neighbor &operator[](size_t i)
    {
        return _data[i];
    }

    Neighbor operator[](size_t i) const
    {
        return _data[i];
    }

    void clear()
    {
        _size = 0;
        _cur = 0;
        _range_filter = std::nullopt;
        _try_new_beamsearch = false;
        _goal_num_filtered = 0;
    }

    bool meets_constraint(uint32_t id) const
    {
        return _meets_constraint.at(id);
    }

  private:
    size_t _size, _capacity, _cur, _goal_num_filtered;
    std::vector<Neighbor> _data;
    bool _try_new_beamsearch = getEnvVar("TRY_NEW_BEAMSEARCH") == "1";
    std::optional<std::pair<float, float>> _range_filter = std::nullopt;
    std::unordered_map<uint32_t, bool> _meets_constraint;

    struct NeighborCompareDistance {
        bool operator()(const Neighbor& n1, const Neighbor& n2) {
            return n1.distance > n2.distance;
        }
    };

    size_t num_locked_in_filtered() const;

    std::priority_queue<Neighbor, std::vector<Neighbor>, NeighborCompareDistance> _extra_nodes;
};

} // namespace diskann
