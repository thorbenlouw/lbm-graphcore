//template<typename T>
//// maybe we arrange ours differently?
//class AccelerateVertex<T>: Vertex {
//public:
//    InOut<Vector<T>> cells;
//    float density;
//    float accel;
//    unsigned nx;
//    unsigned ny;
//    Input<Vector<T>> obstacles;
//
//    bool compute() {
//
//        inline const auto at = [nx](uint16_t x, uint16_t y) -> unsigned {
//            unsigned result = y;
//            return result * nx + x
//        };
//        auto w1 = density * accel / 9.0f;
//        auto w2 = density * accel / 36.0f;
//
//        /* modify the 2nd row of the grid */
//        auto jj = ny - 2;
//
//        for (auto ii = 0u; ii < nx; ii++)
//        {
//            auto idx = at(ii, jj);
//            /* if the cell is not occupied and
//            ** we don't send a negative density */
//            if (!obstacles[idx]
//                && (cells[idx].speeds[3] - w1) > 0.f
//                && (cells[idx].speeds[6] - w2) > 0.f
//                && (cells[idx].speeds[7] - w2) > 0.f)
//            {
//                /* increase 'east-side' densities */
//                cells[idx].speeds[1] += w1;
//                cells[idx].speeds[5] += w2;
//                cells[idx].speeds[8] += w2;
//                /* decrease 'west-side' densities */
//                cells[idx].speeds[3] -= w1;
//                cells[idx].speeds[6] -= w2;
//                cells[idx].speeds[7] -= w2;
//            }
//        }
//
//        return true;
//    }
//};