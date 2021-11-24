import React from "react";
import { shallow } from "enzyme";
import BlockDetails from "./BlockDetails";

describe('<BlockDetails />', () => {
    it('test one block deatils', () => {
        const wrapper = shallow(
            <BlockDetails
              data={[{
                attributes:{
                    data:'test'
                }
              }]}
            />
        );
        expect(wrapper).toMatchSnapshot();
        // console.log(wrapper.debug())
    });
});