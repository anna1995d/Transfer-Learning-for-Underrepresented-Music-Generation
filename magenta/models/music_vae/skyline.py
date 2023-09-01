import heapq

def getSkyline(buildings):
      # Stores the building information in the following manner:[left,right,height]
    buildings=list(map(lambda x: [x[0],x[2],x[1]],buildings))
     
    buildings_start=[0] # priority queue
    buildings_end=dict() #map
     
    # Stores the position and height of the present building and whether it is the endpoint of a building
    new_buildings=[]
    for s,e,h in buildings:
        new_buildings.append((s,h,False))
        new_buildings.append((e,h,True))
         
    # Sorting the buildings according to their position
    new_buildings.sort(key= lambda x:(x[0],x[2]))
     
    # Stores the answer
    skyline=[]
    for x,y,end in new_buildings:           
        if not end:
        # if it is the starting point of a building push it in the heap
            if (not skyline) or y>skyline[-1][1]:
                if skyline and x==skyline[-1][0]:
                    skyline[-1][1]=y
                else:
                    skyline.append([x,y])
                heapq.heappush(buildings_start,-y)
            else:
                heapq.heappush(buildings_start,-y)
        else:
        # if it is the ending point of a building
            if y==skyline[-1][1]:
                heapq.heappop(buildings_start)
                if x==skyline[-1][0]:
                    skyline.pop()
                y=heapq.heappop(buildings_start)
                while -y in buildings_end:
                    buildings_end[-y]-=1
                    if buildings_end[-y]==0:
                        del(buildings_end[-y])
                    y=heapq.heappop(buildings_start)
                if -y!=skyline[-1][1]:
                    skyline.append([x,-y])
                heapq.heappush(buildings_start,y)
            else:
                buildings_end[y]=buildings_end.get(y,0)+1
    return skyline
 
buildings = [ [ 1, 11, 5 ], [ 2, 6, 7 ],
  [ 3, 13, 9 ], [ 12, 7, 16 ],
  [ 14, 3, 25 ], [ 19, 18, 22 ],
  [ 23, 13, 29 ], [ 24, 4, 28 ] ]
  
print(getSkyline(buildings))