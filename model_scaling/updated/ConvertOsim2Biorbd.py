# coding: utf-8

from lxml import etree
import inspect
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R_scipy

from .utils import  *
# from .Muscle_editor import *

class ConvertedFromOsim2Biorbd3:
    def __init__(self, path, originfile, version=3):

        self.path = path
        self.originfile = originfile
        self.version = str(version)

        self.data_origin = etree.parse(self.originfile)
        self.root = self.data_origin.getroot()

        self.file = open(self.path, 'w')
        self.file.write('version ' + self.version + '\n')
        self.file.write('\n// File extracted from ' + self.originfile)
        self.file.write('\n')

        def new_text(element):
            if type(element) == str:
                return element
            else:
                return element.text

        def body_list(_self):
            L = []
            for _body in _self.data_origin.xpath(
                    '/OpenSimDocument/Model/BodySet/objects/Body'):
                L.append(_body.get("name"))
            return L

        def parent_body(_body, _late_body):
            ref = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'parent_body'))
            if ref == 'None':
                return _late_body
            else:
                return ref

        def matrix_inertia(_body):
            ref = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia_xx'))
            if ref == 'None':
                _inertia_str = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia'))
                _inertia = [float(s) for s in _inertia_str.split(' ')]
                return _inertia
            else:
                return [ref,
                        new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia_yy')),
                        new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia_zz')),
                        new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia_xy')),
                        new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia_xz')),
                        new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia_yz'))]

        def muscle_list(_self):
            _list = []
            for _muscle in _self.data_origin.xpath(
                    '/OpenSimDocument/Model/ForceSet/objects/Thelen2003Muscle'):
                _list.append(_muscle.get("name"))
            return _list

        def list_pathpoint_muscle(_muscle):
            # return list of viapoint for each muscle
            _viapoint = []
            # TODO warning for other type of pathpoint
            index_pathpoint = index_go_to(go_to(self.root, 'Thelen2003Muscle', 'name', _muscle), 'PathPoint')
            list_index = list(index_pathpoint)
            tronc_list_index = list_index[:len(list_index) - 2]
            tronc_index = ''.join(tronc_list_index)
            index_root = index_go_to(self.root, 'Thelen2003Muscle', 'name', _muscle)
            index_tronc_total = index_root + tronc_index
            i = 0
            while True:
                try:
                    child = eval('self.root' + index_tronc_total + str(i) + ']')
                    _viapoint.append(child.get("name"))
                    i += 1
                except:  # Exception as e:   print('Error', e)
                    break
            return _viapoint

        def list_transform_body(_body):
            # return list of transformation for each body
            _translation = []
            _rotation = []
            index_transformation = index_go_to(go_to(self.root, 'Body', 'name', _body), 'TransformAxis')
            print(index_transformation, _body)
            if index_transformation is None:
                return [[], []]
            else:
                list_index = list(index_transformation)
                tronc_list_index = list_index[:len(list_index) - 2]
                tronc_index = ''.join(tronc_list_index)
                index_root = index_go_to(self.root, 'Body', 'name', _body)
                index_tronc_total = index_root + tronc_index
                i = 0
                while True:
                    try:
                        child = eval('self.root' + index_tronc_total + str(i) + ']')
                        if child.get('name') is not None:
                            _translation.append(child.get("name")) if child.get('name').find(
                                'translation') == 0 else True
                            _rotation.append(child.get("name")) if child.get('name').find('rotation') == 0 else True
                        i += 1
                    except:  # Exception as e:  print('Error', e)
                        break
                return [_translation, _rotation]

        def list_markers_body(_body):
            # return list of transformation for each body
            markers = []
            index_markers = index_go_to(self.root, 'Marker')
            if index_markers is None:
                return []
            else:
                list_index = list(index_markers)
                tronc_list_index = list_index[:len(list_index) - 2]
                tronc_index = ''.join(tronc_list_index)
                i = 0
                while True:
                    try:
                        child = eval('self.root' + tronc_index + str(i) + ']').get('name')
                        which_body = new_text(go_to(go_to(self.root, 'Marker', 'name', child), 'body'))
                        if which_body == _body:
                            markers.append(child) if child is not None else True
                        i += 1
                    except:  # Exception as e:  print('Error', e)
                        break
                return markers

        def list_dof_body(_body):
            # return list of generalizes coordinates for given body
            dof = []
            index_markers = index_go_to(go_to(self.root, 'Body', 'name', _body), 'Coordinate')
            if index_markers is None:
                return []
            else:
                list_index = list(index_markers)
                tronc_list_index = list_index[:len(list_index) - 2]
                tronc_index = ''.join(tronc_list_index)
                index_root = index_go_to(self.root, 'Body', 'name', _body)
                index_tronc_total = index_root + tronc_index
                i = 0
                while True:
                    try:
                        new_dof = eval('self.root' + index_tronc_total + str(i) + ']').get('name')
                        dof.append(new_dof)
                        i += 1
                    except:  # Exception as e:  print('Error', e)
                        break
                return dof

        def get_body_pathpoint(_pathpoint):
            while True:
                try:
                    if index_go_to(self.root, 'PathPoint', 'name', _pathpoint) is not None or '':
                        if index_go_to(go_to(self.root, 'PathPoint', 'name', _pathpoint), 'body') is not None or '':
                            return new_text(go_to(go_to(self.root, 'PathPoint', 'name', _pathpoint), 'body'))
                        # opensim version 4.0
                        if index_go_to(go_to(self.root, 'PathPoint', 'name', _pathpoint),
                                       'socket_parent_frame') is not None or '':
                            _ref = new_text(go_to(
                                go_to(self.root, 'PathPoint', 'name', _pathpoint), 'socket_parent_frame'))
                            return _ref[9:]
                    elif index_go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint) != '':
                        if index_go_to(go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint), 'body') != '':
                            return new_text(go_to(go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint), 'body'))
                        # opensim version 4.0
                        if index_go_to(go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint),
                                       'socket_parent_frame') is not None or '':
                            _ref = new_text(go_to(
                                go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint), 'socket_parent_frame'))
                            return _ref[9:]
                    elif index_go_to(self.root, 'MovingPathPoint', 'name', _pathpoint) != '':
                        if index_go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint), 'body') != '':
                            return new_text(go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint), 'body'))
                        # opensim version 4.0
                        if index_go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint),
                                       'socket_parent_frame') is not None or '':
                            _ref = new_text(
                                go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint), 'socket_parent_frame'))
                            return _ref[9:]
                    else:
                        return 'None'
                except Exception as e:
                    break

        def get_pos(_pathpoint):
            while True:
                try:
                    if index_go_to(go_to(self.root, 'PathPoint', 'name', _pathpoint), 'location') != '':
                        return new_text(go_to(go_to(self.root, 'PathPoint', 'name', _pathpoint), 'location'))
                    elif index_go_to(go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint), 'location') != '':
                        return new_text(go_to(go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint), 'location'))
                    elif index_go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint), 'location') != '':
                        return new_text(go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint), 'location'))
                    else:
                        return 'None'
                except Exception as e:
                    break

        def muscle_group_reference(_muscle, ref_group):
            for el in ref_group:
                if _muscle == el[0]:
                    return el[1]
            else:
                return 'None'

        # Segment definition
        body_list_actuated = []
        self.write('\n// SEGMENT DEFINITION\n')

        def printing_segment(_body, _name, parent_name, _rotomatrix, transformation_type='', _is_dof='None',
                             true_segment=False, dof_total_trans=''):
            rt_in_matrix = 1
            [[r11, r12, r13, r14],
             [r21, r22, r23, r24],
             [r31, r32, r33, r34],
             [r41, r42, r43, r44]] = _rotomatrix.get_matrix().tolist()
            [i11, i22, i33, i12, i13, i23] = matrix_inertia(_body)
            mass = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'mass'))
            com = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'mass_center'))
            path_mesh_file = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'mesh_file'))
            # TODO add mesh files

            # writing data
            self.write('    // Segment\n')
            self.write('    segment {}\n'.format(_name)) if _name != 'None' else self.write('')
            self.write('        parent {} \n'.format(parent_name)) if parent_name != 'None' else self.write('')
            self.write('        RTinMatrix    {}\n'.format(rt_in_matrix)) if rt_in_matrix != 'None' else self.write('')
            self.write('        RT\n')
            self.write(
                '            {}    {}    {}    {}\n'
                '            {}    {}    {}    {}\n'
                '            {}    {}    {}    {}\n'
                '            {}    {}    {}    {}\n'
                    .format(r11, r12, r13, r14,
                            r21, r22, r23, r24,
                            r31, r32, r33, r34,
                            r41, r42, r43, r44))
            self.write('        translations {}\n'.format(
                dof_total_trans)) if transformation_type == 'translation' and dof_total_trans != '' else True
            self.write('        rotations {}\n'.format('z')) if _is_dof == 'True' else True
            self.write('        mass {}\n'.format(mass)) if true_segment is True else True
            self.write('        inertia\n'
                       '            {}    {}    {}\n'
                       '            {}    {}    {}\n'
                       '            {}    {}    {}\n'
                       .format(i11, i12, i13,
                               i12, i22, i23,
                               i13, i23, i33)) if true_segment is True else True
            self.write('        com    {}\n'.format(com)) if true_segment is True else True
            self.write('        //meshfile {}\n'.format(path_mesh_file)) if path_mesh_file != 'None' else True
            self.write('    endsegment\n')

        # Division of body in segment depending of transformation
        late_body = 'None'
        for body in body_list(self):
            rotomatrix = OrthoMatrix([0, 0, 0])
            self.write('\n// Information about {} segment\n'.format(body))
            parent = parent_body(body, late_body)
            list_transform = list_transform_body(body)
            rotation_for_markers = rotomatrix.get_rotation_matrix()
            # segment data
            if list_transform[0] == []:
                if list_transform[1] == []:
                    printing_segment(body, body, parent, rotomatrix, true_segment=True)
                    body_list_actuated.append(body)
                    parent = body
            else:
                body_trans = body + '_translation'
                dof_total_trans = ''
                j = 0
                list_trans_dof = ['x', 'y', 'z']
                for translation in list_transform[0]:
                    if translation.find('translation') == 0:
                        axis_str = new_text(
                            go_to(go_to(go_to(self.root, 'Body', 'name', body), 'TransformAxis', 'name', translation),
                                  'axis'))
                        axis = [float(s) for s in axis_str.split(' ')]
                        rotomatrix.product(OrthoMatrix([0, 0, 0], axis))
                        is_dof = new_text(
                            go_to(go_to(go_to(self.root, 'Body', 'name', body), 'TransformAxis', 'name', translation),
                                  'coordinates'))
                        if is_dof in list_dof_body(body):
                            dof_total_trans += list_trans_dof[j]
                    j += 1
                trans_str = new_text(go_to(go_to(self.root, 'Body', 'name', body), 'location_in_parent'))
                trans_value = []
                for s in trans_str.split(' '):
                    if s != '':
                        trans_value.append(float(s))
                rotomatrix.product(OrthoMatrix(trans_value))
                rotation_for_markers = rotomatrix.get_rotation_matrix()
                if list_transform[1] == []:
                    is_true_segment = True
                else:
                    is_true_segment = False
                printing_segment(body, body_trans, parent, rotomatrix, 'translation', dof_total_trans,
                                 true_segment=is_true_segment)
                # parent = body_trans
            if list_transform[1] != []:
                rotomatrix = OrthoMatrix([0, 0, 0])
                for rotation in list_transform[1]:
                    if rotation.find('rotation') == 0:
                        axis_str = new_text(
                            go_to(go_to(go_to(self.root, 'Body', 'name', body), 'TransformAxis', 'name', rotation),
                                  'axis'))
                        axis = [float(s) for s in axis_str.split(' ')]
                        rotation_axis = rotomatrix.get_axis()
                        if rotation_axis == '':
                            rotation_axis = 'z'
                        rotomatrix = OrthoMatrix([0, 0, 0], axis)
                        is_dof = new_text(
                            go_to(go_to(go_to(self.root, 'Body', 'name', body), 'TransformAxis', 'name', rotation),
                                  'coordinates'))
                        if is_dof in list_dof_body(body):
                            is_dof = 'True'
                        else:
                            is_dof = 'None'
                        printing_segment(body, body + '_' + rotation, parent, rotomatrix, 'rotation', is_dof)
                        rotation_for_markers = rotation_for_markers.dot(rotomatrix.get_rotation_matrix())
                        parent = body + '_' + rotation

                # segment to cancel axis effects
                rotomatrix.set_rotation_matrix(inv(rotation_for_markers))
                printing_segment(body, body, parent, rotomatrix, true_segment=True)
                parent = body

            # Markers
            _list_markers = list_markers_body(body)
            if _list_markers is not []:
                self.write('\n    // Markers')
                for marker in _list_markers:
                    position = new_text(go_to(go_to(self.root, 'Marker', 'name', marker), 'location'))
                    self.write('\n    marker    {}'.format(marker))
                    self.write('\n        parent    {}'.format(parent))
                    self.write('\n        position    {}'.format(position))
                    self.write('\n    endmarker\n')
            late_body = body

        # Muscle definition
        self.write('\n// MUSCLE DEFINIION\n')
        sort_muscle = []
        muscle_ref_group = []
        for muscle in muscle_list(self):
            viapoint = list_pathpoint_muscle(muscle)
            bodies_viapoint = []
            for pathpoint in viapoint:
                bodies_viapoint.append(get_body_pathpoint(pathpoint))
            # it is supposed that viapoints are organized in order
            # from the parent body to the child body
            body_start = bodies_viapoint[0]
            body_end = bodies_viapoint[len(bodies_viapoint) - 1]
            sort_muscle.append([body_start, body_end])
            muscle_ref_group.append([muscle, body_start + '_to_' + body_end])
        # selecting muscle group
        group_muscle = []
        for ext_muscle in sort_muscle:
            if ext_muscle not in group_muscle:
                group_muscle.append(ext_muscle)
                # print muscle group
        for muscle_group in group_muscle:
            self.write('\n// {} > {}\n'.format(muscle_group[0], muscle_group[1]))
            self.write('musclegroup {}\n'.format(muscle_group[0] + '_to_' + muscle_group[1]))
            self.write('    OriginParent        {}\n'.format(muscle_group[0]))
            self.write('    InsertionParent        {}\n'.format(muscle_group[1]))
            self.write('endmusclegroup\n')
            # muscle
            for muscle in muscle_list(self):
                # muscle data
                m_ref = muscle_group_reference(muscle, muscle_ref_group)
                if m_ref == muscle_group[0] + '_to_' + muscle_group[1]:
                    muscle_type = 'hillthelen'
                    state_type = 'buchanan'
                    start_point = list_pathpoint_muscle(muscle)[0]
                    end_point = list_pathpoint_muscle(muscle)[len(list_pathpoint_muscle(muscle)) - 1]
                    start_pos = get_pos(start_point)
                    insert_pos = get_pos(end_point)
                    opt_length = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'optimal_fiber_length'))
                    max_force = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'max_isometric_force'))
                    tendon_slack_length = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'tendon_slack_length'))
                    pennation_angle = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'pennation_angle_at_optimal'))
                    pcsa = new_text(go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'pcsa'))
                    max_velocity = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'max_contraction_velocity'))

                    # print muscle data
                    self.write('\n    muscle    {}'.format(muscle))
                    self.write('\n        Type    {}'.format(muscle_type)) if muscle_type != 'None' else self.write('')
                    self.write('\n        statetype    {}'.format(state_type)) if state_type != 'None' else self.write(
                        '')
                    self.write('\n        musclegroup    {}'.format(m_ref)) if m_ref != 'None' else self.write('')
                    self.write(
                        '\n        OriginPosition    {}'.format(start_pos)) if start_pos != 'None' else self.write('')
                    self.write(
                        '\n        InsertionPosition    {}'.format(insert_pos)) if insert_pos != 'None' else self.write(
                        '')
                    self.write(
                        '\n        optimalLength    {}'.format(opt_length)) if opt_length != 'None' else self.write('')
                    self.write('\n        maximalForce    {}'.format(max_force)) if max_force != 'None' else self.write(
                        '')
                    self.write('\n        tendonSlackLength    {}'.format(
                        tendon_slack_length)) if tendon_slack_length != 'None' else self.write('')
                    self.write('\n        pennationAngle    {}'.format(
                        pennation_angle)) if pennation_angle != 'None' else self.write('')
                    self.write('\n        PCSA    {}'.format(pcsa)) if pcsa != 'None' else self.write('')
                    self.write(
                        '\n        maxVelocity    {}'.format(max_velocity)) if max_velocity != 'None' else self.write(
                        '')
                    self.write('\n    endmuscle\n')
                    # viapoint
                    for viapoint in list_pathpoint_muscle(muscle):
                        # viapoint data
                        parent_viapoint = get_body_pathpoint(viapoint)
                        viapoint_pos = get_pos(viapoint)
                        # print viapoint data
                        self.write('\n        viapoint    {}'.format(viapoint))
                        self.write('\n            parent    {}'.format(
                            parent_viapoint)) if parent_viapoint != 'None' else self.write('')
                        self.write('\n            muscle    {}'.format(muscle))
                        self.write('\n            musclegroup    {}'.format(m_ref)) if m_ref != 'None' else self.write(
                            '')
                        self.write('\n            position    {}'.format(
                            viapoint_pos)) if viapoint_pos != 'None' else self.write('')
                        self.write('\n        endviapoint')
                    self.write('\n')

        self.file.close()

    def __getattr__(self, attr):
        print('Error : {} is not an attribute of this class'.format(attr))

    def get_path(self):
        return self.path

    def write(self, string):
        self.file = open(self.path, 'a')
        self.file.write(string)
        self.file.close()

    def get_origin_file(self):
        return self.originfile

    def credits(self):
        return self.data_origin.xpath(
            '/OpenSimDocument/Model/credits')[0].text

    def publications(self):
        return self.data_origin.xpath(
            '/OpenSimDocument/Model/publications')[0].text

    def body_list(self):
        _list = []
        for body in self.data_origin.xpath(
                '/OpenSimDocument/Model/BodySet/objects/Body'):
            _list.append(body.get("name"))
        return _list

class ConvertedFromOsim2Biorbd4:
    def __init__(self, path, origin_file, version=3):

        self.path = path
        self.origin_file = origin_file
        self.version = str(version)

        self.data_origin = etree.parse(self.origin_file)
        self.root = self.data_origin.getroot()

        self.file = open(self.path, 'w')
        self.file.write('version ' + self.version + '\n')
        self.file.write('\n// File extracted from ' + self.origin_file)
        self.file.write('\n')

        def new_text(element):
            if type(element) == str:
                return element
            else:
                return element.text

        def body_list(_self):
            list_of_bodies = []
            for _body in _self.data_origin.xpath(
                    '/OpenSimDocument/Model/BodySet/objects/Body'):
                list_of_bodies.append(_body.get("name"))
            return list_of_bodies

        def matrix_inertia(_body):
            _ref = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia'))
            if _ref != 'None':
                _inertia_str = _ref
                _inertia = [float(s) for s in _inertia_str.split(' ')]
                return _inertia
            else:
                return 'None'

        def muscle_list(_self):
            _list = []
            for _muscle in _self.data_origin.xpath(
                    '/OpenSimDocument/Model/ForceSet/objects/Thelen2003Muscle'):
                _list.append(_muscle.get("name"))
            return _list

        def list_pathpoint_muscle(_muscle):
            # return list of viapoint for each muscle
            _viapoint = []
            # TODO warning for other type of pathpoint
            index_pathpoint = index_go_to(go_to(self.root, 'Thelen2003Muscle', 'name', _muscle), 'PathPoint')
            _list_index = list(index_pathpoint)
            _tronc_list_index = _list_index[:len(_list_index) - 2]
            _tronc_index = ''.join(_tronc_list_index)
            index_root = index_go_to(self.root, 'Thelen2003Muscle', 'name', _muscle)
            index_tronc_total = index_root + _tronc_index
            i = 0
            while True:
                try:
                    child = eval('self.root' + index_tronc_total + str(i) + ']')
                    _viapoint.append(child.get("name"))
                    i += 1
                except:  # Exception as e:   print('Error', e)
                    break
            return _viapoint

        def list_markers_body(_body):
            # return list of transformation for each body
            markers = []
            index_markers = index_go_to(self.root, 'Marker')
            if index_markers is None:
                return []
            else:
                _list_index = list(index_markers)
                _tronc_list_index = _list_index[:len(_list_index) - 2]
                _tronc_index = ''.join(_tronc_list_index)
                i = 0
                while True:
                    try:
                        child = eval('self.root' + _tronc_index + str(i) + ']').get('name')
                        which_body = new_text(go_to(go_to(self.root, 'Marker', 'name', child), 'socket_parent_frame'))[
                                     9:]
                        if which_body == _body:
                            markers.append(child) if child is not None else True
                        i += 1
                    except:
                        break
                return markers

        # list of joints with parent and child
        list_joint = []
        index_joints = index_go_to(self.root, 'WeldJoint')
        if index_joints is not None:
            list_index = list(index_joints)
            tronc_list_index = list_index[:len(list_index) - 2]
            tronc_index = ''.join(tronc_list_index)
            i = 0
            while True:
                try:
                    new_joint = eval('self.root' + tronc_index + str(i) + ']').get('name')
                    if new_text(go_to(self.root, 'WeldJoint', 'name', new_joint)) != 'None':
                        _parent_joint = new_text(
                            go_to(go_to(self.root, 'WeldJoint', 'name', new_joint), 'socket_parent_frame'))[:-7]
                        _child_joint = new_text(
                            go_to(go_to(self.root, 'WeldJoint', 'name', new_joint), 'socket_child_frame'))[:-7]
                        list_joint.append([new_joint, _parent_joint, _child_joint, 'WeldJoint'])
                    i += 1
                except:  # Exception as error:
                    # print('Error', error)
                    break
        index_joints = index_go_to(self.root, 'CustomJoint')
        if index_joints is not None:
            list_index = list(index_joints)
            tronc_list_index = list_index[:len(list_index) - 2]
            tronc_index = ''.join(tronc_list_index)
            i = int(list_index[len(list_index) - 2])
            while True:
                try:
                    new_joint = eval('self.root' + tronc_index + str(i) + ']').get('name')
                    if new_text(go_to(self.root, 'CustomJoint', 'name', new_joint)) != 'None':
                        _parent_joint = new_text(
                            go_to(go_to(self.root, 'CustomJoint', 'name', new_joint), 'socket_parent_frame'))[:-7]
                        _child_joint = new_text(
                            go_to(go_to(self.root, 'CustomJoint', 'name', new_joint), 'socket_child_frame'))[:-7]
                        list_joint.append([new_joint, _parent_joint, _child_joint, 'CustomJoint'])
                    i += 1
                except:  # Exception as e:print('Error', e)
                    break

        def dof_of_joint(_joint, _joint_type):
            dof = []
            _index_dof = index_go_to(go_to(self.root, _joint_type, 'name', _joint), 'Coordinate')
            if _index_dof is None:
                return []
            else:
                _list_index = list(_index_dof)
                _tronc_list_index = _list_index[:len(_list_index) - 2]
                _tronc_index = ''.join(_tronc_list_index)
                _index_root = index_go_to(self.root, _joint_type, 'name', _joint)
                _index_tronc_total = _index_root + _tronc_index
                i = 0
                while True:
                    try:
                        child = eval('self.root' + _index_tronc_total + str(i) + ']')
                        if child.get('name') is not None:
                            dof.append(child.get("name"))
                        i += 1
                    except:
                        break
            return dof

        def parent_child(_child):
            # return parent of a child
            # suppose that a parent can only have one child
            for _joint in list_joint:
                if _joint[2] == _child:
                    return _joint[1]
            else:
                return 'None'

        def joint_body(_body):
            # return the joint to which the body is child
            for _joint in list_joint:
                if _joint[2] == _body:
                    return _joint[0], _joint[3]
            else:
                return 'None', 'None'

        def transform_of_joint(_joint, _joint_type):
            _translation = []
            _rotation = []
            if _joint is 'None':
                return [[], []]
            _index_transform = index_go_to(go_to(self.root, _joint_type, 'name', _joint), 'TransformAxis')
            if _index_transform is None:
                return [[], []]
            else:
                _list_index = list(_index_transform)
                _tronc_list_index = _list_index[:len(_list_index) - 2]
                _tronc_index = ''.join(_tronc_list_index)
                _index_root = index_go_to(self.root, _joint_type, 'name', _joint)
                if not _index_root:
                    pass
                _index_tronc_total = _index_root + _tronc_index
                i = 0
                while True:
                    try:
                        child = eval('self.root' + _index_tronc_total + str(i) + ']')
                        if child.get('name') is not None:
                            _translation.append(child.get("name")) \
                                if child.get('name').find('translation') == 0 else True
                            _rotation.append(child.get("name")) \
                                if child.get('name').find('rotation') == 0 else True
                        i += 1
                    except:  # Exception as e:  print('Error', e)
                        break
            return [_translation, _rotation]

        def get_body_pathpoint(_pathpoint):
            while True:
                try:
                    if index_go_to(go_to(self.root, 'PathPoint', 'name', _pathpoint),
                                   'socket_parent_frame') is not None or '':
                        _ref = new_text(go_to(
                            go_to(self.root, 'PathPoint', 'name', _pathpoint), 'socket_parent_frame'))
                        return _ref[9:]
                    if index_go_to(go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint),
                                   'socket_parent_frame') is not None or '':
                        _ref = new_text(go_to(
                            go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint), 'socket_parent_frame'))
                        return _ref[9:]
                    if index_go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint),
                                   'socket_parent_frame') is not None or '':
                        _ref = new_text(
                            go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint), 'socket_parent_frame'))
                        return _ref[9:]
                    else:
                        return 'None'
                except Exception as e:
                    break

        def get_pos(_pathpoint):
            while True:
                try:
                    if index_go_to(go_to(self.root, 'PathPoint', 'name', _pathpoint), 'location') != '':
                        return new_text(go_to(go_to(self.root, 'PathPoint', 'name', _pathpoint), 'location'))
                    elif index_go_to(go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint), 'location') != '':
                        return new_text(go_to(go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint), 'location'))
                    elif index_go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint), 'location') != '':
                        return new_text(go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint), 'location'))
                    else:
                        return 'None'
                except Exception as e:
                    break

        def muscle_group_reference(_muscle, ref_group):
            for el in ref_group:
                if _muscle == el[0]:
                    return el[1]
            else:
                return 'None'

        #        # Credits
        #        self.write('\n// CREDITS')
        #        _credits = print_credits()
        #        self.write('\n'+_credits+'\n')
        #
        #         # Publications
        #        self.write('\n// PUBLICATIONS\n')
        #        _publications = print_publications()
        #        self.write('\n'+_publications+'\n')

        # Segment definition
        self.write('\n// SEGMENT DEFINITION\n')

        # TODO change spaces into \t
        def printing_segment(_body, _name, parent_name, _rotomatrix, transformation_type='', _is_dof='None',
                             true_segment=False, _dof_total_trans=''):
            rt_in_matrix = 1
            [[r11, r12, r13, r14],
             [r21, r22, r23, r24],
             [r31, r32, r33, r34],
             [r41, r42, r43, r44]] = _rotomatrix.get_matrix().tolist()
            for i in range(4):
                for j in range(4):
                    round(eval('r' + str(i + 1) + str(j + 1)), 9)
            [i11, i22, i33, i12, i13, i23] = matrix_inertia(_body)
            mass = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'mass'))
            com = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'mass_center'))
            path_mesh_file = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'mesh_file'))
            # TODO add mesh files

            # writing data
            self.write('    // Segment\n')
            self.write('    segment {}\n'.format(_name)) if _name != 'None' else self.write('')
            self.write('        parent {} \n'.format(parent_name)) if parent_name != 'None' else self.write('')
            self.write('        RTinMatrix    {}\n'.format(rt_in_matrix)) if rt_in_matrix != 'None' else self.write('')
            self.write('        RT\n')
            self.write(
                '            {}    {}    {}    {}\n'
                '            {}    {}    {}    {}\n'
                '            {}    {}    {}    {}\n'
                '            {}    {}    {}    {}\n'
                    .format(r11, r12, r13, r14,
                            r21, r22, r23, r24,
                            r31, r32, r33, r34,
                            r41, r42, r43, r44))
            self.write('        translations {}\n'.format(
                _dof_total_trans)) if transformation_type == 'translation' and _dof_total_trans != '' else True
            self.write('        rotations {}\n'.format('z')) if _is_dof == 'True' else True
            self.write('        mass {}\n'.format(mass)) if true_segment is True else True
            self.write('        inertia\n'
                       '            {}    {}    {}\n'
                       '            {}    {}    {}\n'
                       '            {}    {}    {}\n'
                       .format(i11, i12, i13,
                               i12, i22, i23,
                               i13, i23, i33)) if true_segment is True else True
            self.write('        com    {}\n'.format(com)) if true_segment is True else True
            self.write('        //meshfile {}\n'.format(path_mesh_file)) if path_mesh_file != 'None' else True
            self.write('    endsegment\n')

        # Division of body in segment depending of transformation
        for body in body_list(self):
            rotomatrix = OrthoMatrix([0, 0, 0])
            self.write('\n// Information about {} segment\n'.format(body))
            parent = parent_child(body)
            # if parent == 'ground':
            #     parent = 'None'
            joint, joint_type = joint_body(body)
            list_transform = transform_of_joint(joint, joint_type)
            rotation_for_markers = rotomatrix.get_rotation_matrix()
            # segment data
            if list_transform[0] == []:
                if list_transform[1] == []:
                    printing_segment(body, body, parent, rotomatrix, true_segment=True)
                    parent = body
            else:
                body_trans = body + '_translation'
                dof_total_trans = ''
                j = 0
                list_trans_dof = ['x', 'y', 'z']
                for translation in list_transform[0]:
                    if translation.find('translation') == 0:
                        axis_str = new_text(go_to(
                            go_to(go_to(self.root, joint_type, 'name', joint), 'TransformAxis', 'name', translation),
                            'axis'))
                        axis = [float(s) for s in axis_str.split(' ')]
                        rotomatrix.product(OrthoMatrix([0, 0, 0], axis))
                        is_dof = new_text(go_to(
                            go_to(go_to(self.root, joint_type, 'name', joint), 'TransformAxis', 'name', translation),
                            'coordinates'))
                        if is_dof in dof_of_joint(joint, joint_type):
                            dof_total_trans += list_trans_dof[j]
                    j += 1
                trans_str = new_text(go_to(
                    go_to(go_to(self.root, joint_type, 'name', joint), 'PhysicalOffsetFrame', 'name',
                          parent + '_offset'), 'translation'))
                trans_value = []
                for s in trans_str.split(' '):
                    if s != '' and s is not 'None':
                        trans_value.append(float(s))
                rotomatrix.product(OrthoMatrix(trans_value))
                rotation_for_markers = rotomatrix.get_rotation_matrix()


                if list_transform[1] == []:
                    is_true_segment = True
                else:
                    is_true_segment = False
                printing_segment(body, body_trans, parent, rotomatrix, 'translation', dof_total_trans,
                                 true_segment=is_true_segment)
                parent = body_trans
            if list_transform[1] != []:
                rotomatrix = OrthoMatrix([0, 0, 0])
                for rotation in list_transform[1]:
                    if rotation.find('rotation') == 0:
                        axis_str = new_text(
                            go_to(go_to(go_to(self.root, joint_type, 'name', joint), 'TransformAxis', 'name', rotation),
                                  'axis'))
                        axis = [float(s) for s in axis_str.split(' ')]
                        rotomatrix = OrthoMatrix([0, 0, 0], axis)
                        is_dof = new_text(
                            go_to(go_to(go_to(self.root, joint_type, 'name', joint), 'TransformAxis', 'name', rotation),
                                  'coordinates'))
                        if is_dof in dof_of_joint(joint, joint_type):
                            is_dof = 'True'
                        else:
                            is_dof = 'None'
                        printing_segment(body, body + '_' + rotation, parent, rotomatrix, 'rotation', is_dof)
                        rotation_for_markers = rotation_for_markers.dot(rotomatrix.get_rotation_matrix())
                        parent = body + '_' + rotation

                # segment to cancel axis effects
                rotomatrix.set_rotation_matrix(inv(rotation_for_markers))
                printing_segment(body, body, parent, rotomatrix, true_segment=True)
                parent = body

            # Markers
            _list_markers = list_markers_body(body)
            if _list_markers is not []:
                self.write('\n    // Markers')
                for marker in _list_markers:
                    position = new_text(go_to(go_to(self.root, 'Marker', 'name', marker), 'location'))
                    self.write('\n    marker    {}'.format(marker))
                    self.write('\n        parent    {}'.format(parent))
                    self.write('\n        position    {}'.format(position))
                    self.write('\n    endmarker\n')
            late_body = body

        # Muscle definition
        self.write('\n// MUSCLE DEFINIION\n')
        sort_muscle = []
        muscle_ref_group = []
        for muscle in muscle_list(self):
            viapoint = list_pathpoint_muscle(muscle)
            bodies_viapoint = []
            for pathpoint in viapoint:
                bodies_viapoint.append(get_body_pathpoint(pathpoint))
            # it is supposed that viapoints are organized in order
            # from the parent body to the child body
            body_start = bodies_viapoint[0]
            body_end = bodies_viapoint[len(bodies_viapoint) - 1]
            sort_muscle.append([body_start, body_end])
            muscle_ref_group.append([muscle, body_start + '_to_' + body_end])
        # selecting muscle group
        group_muscle = []
        for ext_muscle in sort_muscle:
            if ext_muscle not in group_muscle:
                group_muscle.append(ext_muscle)
                # print muscle group
        for muscle_group in group_muscle:
            self.write('\n// {} > {}\n'.format(muscle_group[0], muscle_group[1]))
            self.write('musclegroup {}\n'.format(muscle_group[0] + '_to_' + muscle_group[1]))
            self.write('    OriginParent        {}\n'.format(muscle_group[0]))
            self.write('    InsertionParent        {}\n'.format(muscle_group[1]))
            self.write('endmusclegroup\n')
            # muscle
            for muscle in muscle_list(self):
                # muscle data
                m_ref = muscle_group_reference(muscle, muscle_ref_group)
                if m_ref == muscle_group[0] + '_to_' + muscle_group[1]:
                    muscle_type = 'hillthelen'
                    state_type = 'buchanan'
                    list_pathpoint = list_pathpoint_muscle(muscle)
                    start_point = list_pathpoint.pop(0)
                    end_point = list_pathpoint.pop()
                    start_pos = get_pos(start_point)
                    insert_pos = get_pos(end_point)
                    opt_length = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'optimal_fiber_length'))
                    max_force = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'max_isometric_force'))
                    tendon_slack_length = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'tendon_slack_length'))
                    pennation_angle = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'pennation_angle_at_optimal'))
                    pcsa = new_text(go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'pcsa'))
                    max_velocity = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'max_contraction_velocity'))

                    # print muscle data
                    self.write('\n    muscle    {}'.format(muscle))
                    self.write('\n        Type    {}'.format(muscle_type)) if muscle_type != 'None' else self.write('')
                    self.write('\n        statetype    {}'.format(state_type)) if state_type != 'None' else self.write(
                        '')
                    self.write('\n        musclegroup    {}'.format(m_ref)) if m_ref != 'None' else self.write('')
                    self.write(
                        '\n        OriginPosition    {}'.format(start_pos)) if start_pos != 'None' else self.write('')
                    self.write(
                        '\n        InsertionPosition    {}'.format(insert_pos)) if insert_pos != 'None' else self.write(
                        '')
                    self.write(
                        '\n        optimalLength    {}'.format(opt_length)) if opt_length != 'None' else self.write('')
                    self.write('\n        maximalForce    {}'.format(max_force)) if max_force != 'None' else self.write(
                        '')
                    self.write('\n        tendonSlackLength    {}'.format(
                        tendon_slack_length)) if tendon_slack_length != 'None' else self.write('')
                    self.write('\n        pennationAngle    {}'.format(
                        pennation_angle)) if pennation_angle != 'None' else self.write('')
                    self.write('\n        PCSA    {}'.format(pcsa)) if pcsa != 'None' else self.write('')
                    self.write(
                        '\n        maxVelocity    {}'.format(max_velocity)) if max_velocity != 'None' else self.write(
                        '')
                    self.write('\n    endmuscle\n')
                    # viapoint
                    for viapoint in list_pathpoint:
                        # viapoint data
                        parent_viapoint = get_body_pathpoint(viapoint)
                        viapoint_pos = get_pos(viapoint)
                        # print viapoint data
                        self.write('\n        viapoint    {}'.format(viapoint))
                        self.write('\n            parent    {}'.format(
                            parent_viapoint)) if parent_viapoint != 'None' else self.write('')
                        self.write('\n            muscle    {}'.format(muscle))
                        self.write('\n            musclegroup    {}'.format(m_ref)) if m_ref != 'None' else self.write(
                            '')
                        self.write('\n            position    {}'.format(
                            viapoint_pos)) if viapoint_pos != 'None' else self.write('')
                        self.write('\n        endviapoint')
                    self.write('\n')

        self.file.close()

    def __getattr__(self, attr):
        print('Error : {} is not an attribute of this class'.format(attr))

    def get_path(self):
        return self.path

    def write(self, string):
        self.file = open(self.path, 'a')
        self.file.write(string)
        self.file.close()

    def get_origin_file(self):
        return self.originfile

    def credits(self):
        return self.data_origin.xpath(
            '/OpenSimDocument/Model/credits')[0].text

    def publications(self):
        return self.data_origin.xpath(
            '/OpenSimDocument/Model/publications')[0].text

    def body_list(self):
        _list = []
        for body in self.data_origin.xpath(
                '/OpenSimDocument/Model/BodySet/objects/Body'):
            _list.append(body.get("name"))
        return _list

class ConvertedFromOsim2Biorbd5:
    def __init__(self, path, origin_file, version=3):

        self.path = path
        self.origin_file = origin_file
        self.version = str(version)

        self.data_origin = etree.parse(self.origin_file)
        self.root = self.data_origin.getroot()

        self.file = open(self.path, 'w')
        self.file.write('version ' + self.version + '\n')
        self.file.write('\n// File extracted from ' + self.origin_file)
        self.file.write('\n')
        constraints_in_model = False

        def body_list(_self):
            list_of_bodies = []
            for _body in _self.data_origin.xpath(
                    '/OpenSimDocument/Model/BodySet/objects/Body'):
                list_of_bodies.append(_body.get("name"))
            return list_of_bodies

        def matrix_inertia(_body):
            _ref = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia'))
            if _ref != 'None':
                _inertia_str = _ref
                _inertia = [float(s) for s in _inertia_str.split(' ')]
                return _inertia
            else:
                return 'None'

        ###########    Muscle Related Functions

        def list_pathpoint_muscle(_muscle):
            # return list of via point for each muscle
            _viapoint = []
            # TODO warning for other type of pathpoint
            index_pathpoint = index_go_to(go_to(self.root, 'Thelen2003Muscle', 'name', _muscle), 'PathPoint')
            _list_index = list(index_pathpoint)
            _tronc_list_index = _list_index[:len(_list_index) - 2]
            _tronc_index = ''.join(_tronc_list_index)
            index_root = index_go_to(self.root, 'Thelen2003Muscle', 'name', _muscle)
            index_tronc_total = index_root + _tronc_index
            i = 0
            while True:
                try:
                    child = eval('self.root' + index_tronc_total + str(i) + ']')
                    _viapoint.append(child.get("name"))
                    i += 1
                except:  # Exception as e:   print('Error', e)
                    break
            return _viapoint

        def get_body_pathpoint(_pathpoint):
            while True:
                try:
                    if index_go_to(go_to(self.root, 'PathPoint', 'name', _pathpoint),
                                   'socket_parent_frame') is not None or '':
                        _ref = new_text(go_to(
                            go_to(self.root, 'PathPoint', 'name', _pathpoint), 'socket_parent_frame'))
                        return _ref[9:]
                    if index_go_to(go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint),
                                   'socket_parent_frame') is not None or '':
                        _ref = new_text(go_to(
                            go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint), 'socket_parent_frame'))
                        return _ref[9:]
                    if index_go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint),
                                   'socket_parent_frame') is not None or '':
                        _ref = new_text(
                            go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint), 'socket_parent_frame'))
                        return _ref[9:]
                    else:
                        return 'None'
                except Exception as e:
                    break

        def muscle_list(_self):
            _list = []
            for _muscle in _self.data_origin.xpath(
                    '/OpenSimDocument/Model/ForceSet/objects/Thelen2003Muscle'):
                _list.append(_muscle.get("name"))
            return _list

        def get_pos(_pathpoint):

            while True:
                try:
                    if index_go_to(go_to(self.root, 'PathPoint', 'name', _pathpoint), 'location') != '':
                        return new_text(go_to(go_to(self.root, 'PathPoint', 'name', _pathpoint), 'location'))
                    elif index_go_to(go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint), 'location') != '':
                        return new_text(go_to(go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint), 'location'))
                    elif index_go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint), 'location') != '':
                        return new_text(go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint), 'location'))
                    else:
                        return 'None'
                except Exception as e:
                    break

        def muscle_group_reference(_muscle, ref_group):

            for el in ref_group:
                if _muscle == el[0]:
                    return el[1]
            else:
                return 'None'

        ###########    Markers Related Functions

        def list_markers_body(_body):
            # return list of transformation for each body
            markers = []
            index_markers = index_go_to(self.root, 'Marker')
            if index_markers is None:
                return []
            else:
                _list_index = list(index_markers)
                _tronc_list_index = _list_index[:len(_list_index) - 2]
                _tronc_index = ''.join(_tronc_list_index)
                i = 0
                while True:
                    try:
                        child = eval('self.root' + _tronc_index + str(i) + ']').get('name')
                        which_body = new_text(go_to(go_to(self.root, 'Marker', 'name', child), 'socket_parent_frame'))[
                                     9:]
                        if which_body == _body:
                            markers.append(child) if child is not None else True
                        i += 1
                    except:
                        break
                return markers

        ###########    Joint extraction and other related functions

        def joint_by_type(_joint_type):
            _list_joint_type = []
            _index_joints = index_go_to(self.root, _joint_type)
            if _index_joints is not None:
                _list_index = list(_index_joints)
                _tronc_list_index = _list_index[:len(_list_index) - 2]
                _tronc_index = ''.join(_tronc_list_index)
                i = 0
                # int(list_index[len(list_index) - 2])
                while True:
                    try:
                        _new_joint = eval('self.root' + _tronc_index + str(i) + ']').get('name')
                        if new_text(go_to(self.root, _joint_type, 'name', _new_joint)) != 'None':
                            _parent_joint = new_text(
                                go_to(go_to(self.root, _joint_type, 'name', _new_joint), 'socket_parent_frame'))[:-7]
                            _child_joint = new_text(
                                go_to(go_to(self.root, _joint_type, 'name', _new_joint), 'socket_child_frame'))[:-7]
                            _list_joint_type.append([_new_joint, _parent_joint, _child_joint, _joint_type])
                        i += 1
                    except:  # Exception as error:
                        # print('Error', error)
                        break
            return _list_joint_type

        # list of joints with parent and child
        list_joint = []
        for ijoint_type in ['WeldJoint', 'CustomJoint', 'PinJoint']:
            list_joint.extend(joint_by_type(ijoint_type))

        def dof_of_joint(_joint, _joint_type):
            dof = []
            _index_dof = index_go_to(go_to(self.root, _joint_type, 'name', _joint), 'Coordinate')
            if _index_dof is None:
                return []
            else:
                _list_index = list(_index_dof)
                _tronc_list_index = _list_index[:len(_list_index) - 2]
                _tronc_index = ''.join(_tronc_list_index)
                _index_root = index_go_to(self.root, _joint_type, 'name', _joint)
                _index_tronc_total = _index_root + _tronc_index
                i = 0
                while True:
                    try:
                        child = eval('self.root' + _index_tronc_total + str(i) + ']')
                        if child.get('name') is not None:
                            dof.append(child.get("name"))
                        i += 1
                    except:
                        break
            return dof

        def parent_child(_child):
            # return parent of a child
            # suppose that a parent can only have one child
            for _joint in list_joint:
                if _joint[2] == _child:
                    return _joint[1]
            else:
                return 'None'

        def joint_body(_body):
            # return the joint to which the body is child
            for _joint in list_joint:
                if _joint[2] == _body:
                    return _joint[0], _joint[3]
            else:
                return 'None', 'None'

        def transform_of_joint(_joint, _joint_type):
            _translation = []
            _rotation = []
            if _joint is 'None':
                return [[], []]
            _index_transform = index_go_to(go_to(self.root, _joint_type, 'name', _joint), 'TransformAxis')
            if _index_transform is None:
                return [[], []]
            else:
                _list_index = list(_index_transform)
                _tronc_list_index = _list_index[:len(_list_index) - 2]
                _tronc_index = ''.join(_tronc_list_index)
                _index_root = index_go_to(self.root, _joint_type, 'name', _joint)
                if not _index_root:
                    pass
                _index_tronc_total = _index_root + _tronc_index
                i = 0
                while True:
                    try:
                        child = eval('self.root' + _index_tronc_total + str(i) + ']')
                        if child.get('name') is not None:
                            _translation.append(child.get("name")) \
                                if child.get('name').find('translation') == 0 and is_coordinate_active(child) else True
                            _rotation.append(child.get("name")) \
                                if child.get('name').find('rotation') == 0 and is_coordinate_active(child) else True
                        i += 1
                    except:  # Exception as e:  print('Error', e)
                        break
            return [_translation, _rotation]

        def is_coordinate_active(transform_axis):
            dof_active = False
            for i_att in transform_axis.getchildren():
                if i_att.tag == 'coordinates':
                    if i_att.text:
                        dof_active = True
            return dof_active

        def get_frames_offsets(_joint, _joint_type):
            offset_data = []
            if _joint is 'None':
                return [[], []]

            joint_of_interest = go_to(self.root, _joint_type, 'name', _joint)
            for i_att in joint_of_interest.getchildren():
                info = {}
                if i_att.tag =='frames':
                    for j in i_att.getchildren():
                        for j_info in j.getchildren():
                            if j_info.tag in ['socket_parent', 'translation','orientation']:
                                info.update({j_info.tag:j_info.text})
                        offset_data.append(info.copy())

            # Todo : add check that first is parent and second is child
            offset_parent = [[float(i) for i in offset_data[0]['translation'].split(' ')],
                             [float(i) for i in offset_data[0]['orientation'].split(' ')]]

            offset_child = [[float(i) for i in offset_data[1]['translation'].split(' ')],
                             [-float(i) for i in offset_data[1]['orientation'].split(' ')]]

            if any([item for sublist in offset_child for item in sublist]):
                R = compute_matrix_rotation(offset_child[1]).T
                compare_R = R_scipy.from_euler('xyz',[offset_child[1]]).as_dcm()[0].T
                new_translation = -np.dot(R.T, offset_child[0])
                new_rotation = -rot2eul(R)
                offset_child = [new_translation, new_rotation]
            else:
                offset_child = []

            return [offset_parent, offset_child]

        def get_q_range(_is_dof, _joint_type, _joint):
            range_q = new_text(go_to(go_to(go_to(self.root, _joint_type, 'name', _joint), 'Coordinate', 'name', _is_dof),
                                     'range'))
            range_value = []
            for r in range_q.split(' '):
                if r != '' and r is not 'None':
                    range_value.append(float(r))
            return range_value

        def update_q_range(_range_q, _Transform_function):
            if _Transform_function[0] == 'LinearFunction':
                new_range = [i_r * float(_Transform_function[1][0]) + float(_Transform_function[1][1]) for i_r in _range_q]
            elif _Transform_function[0] == 'SimmSpline':
                y_value = [float(i) for i in ' '.join(_Transform_function[1][1].split(' ')).split()]
                new_range = [min(y_value), max(y_value)]
            elif _Transform_function[0] == 'MultiplierFunction':
                y_value = [float(i)*float(_Transform_function[1]) for i in ' '.join(_Transform_function[2][1].split(' ')).split()]
                new_range = [min(y_value), max(y_value)]
            return new_range

        def extract_information_dof(_list_transform, _type_transf, _joint_type, _joint, range_q):
            dof_chain_loc = ''
            _dof_total_transf = ''
            list_trans_dof = ['x', 'y', 'z']
            for i_transf in _list_transform:
                if i_transf.find(_type_transf) == 0:
                    dof_information = '\n'

                    axis_str = new_text(go_to(
                        go_to(
                            go_to(self.root, _joint_type, 'name', _joint),
                            'TransformAxis', 'name', i_transf)
                        , 'axis'))

                    axis = [float(s) for s in axis_str.split(' ')]

                    is_dof = new_text(go_to(
                        go_to(
                            go_to(self.root, _joint_type, 'name', _joint),
                            'TransformAxis', 'name', i_transf),
                        'coordinates'))

                    Transform_function = []

                    dof_information += f'//dof axis: {_joint} at {i_transf} on {axis_str}\n'

                    for i_att in go_to(go_to(self.root, joint_type, 'name', joint), 'TransformAxis',
                                       'name', i_transf).getchildren():

                        if i_att.tag == 'coordinates':
                            dof_information += f'//coordinates that serve as the independent variables: {i_att.text}\n'

                        if i_att.tag in ['LinearFunction', 'SimmSpline', 'MultiplierFunction']:
                            Transform_function.append(i_att.tag)

                            if i_att.tag == 'LinearFunction':
                                linear_func = ' '.join(i_att[0].text.split(' ')).split()
                                Transform_function.append(linear_func)
                                if linear_func != ['1', '0']:
                                    constraints_in_model = True

                            elif i_att.tag == 'SimmSpline':
                                Transform_function.append([i_att[0].text, i_att[1].text])
                                constraints_in_model = True

                            elif i_att.tag == 'MultiplierFunction':
                                for iparam in i_att.getchildren():
                                    if iparam.tag == 'scale':
                                        scale = iparam.text
                                    if iparam.tag == 'function':
                                        simmsp = iparam[0][0].text #, iparam[0][1].text]
                                Transform_function.append(scale)
                                Transform_function.append(simmsp)
                                constraints_in_model = True

                    dof_information += f'//Transform function: \n\t//Function type: {Transform_function[0]}\n' \
                        f'\t//Parameters: {Transform_function[1:]}\n'

                    if not Transform_function:
                        raise (f'Transform function for {_joint} at {axis_str} is unknown,'
                               f' only LinearFunction,  SimmSpline and MultiplierFunction are implemented')

                    if is_dof in dof_of_joint(joint, joint_type):
                        # _dof_total_transf += list_trans_dof[axis.index(1.0)]
                        range_q_value = get_q_range(is_dof, _joint_type, _joint)
                        range_q_value_updated = update_q_range(range_q_value, Transform_function)
                        range_q.append(range_q_value_updated)
                        if 1.0 in axis:
                            _dof_total_transf += list_trans_dof[axis.index(1.0)]
                        else:
                            _dof_total_transf += list_trans_dof[axis.index(-1.0)]
                    dof_chain_loc += dof_information

            return dof_chain_loc, range_q, _dof_total_transf
        #        # Credits
        #        self.write('\n// CREDITS')
        #        _credits = print_credits()
        #        self.write('\n'+_credits+'\n')
        #
        #         # Publications
        #        self.write('\n// PUBLICATIONS\n')
        #        _publications = print_publications()
        #        self.write('\n'+_publications+'\n')

        ###########    Segment definition

        self.write('\n// SEGMENT DEFINITION\n')
        # TODO change spaces into \t
        def printing_segment(_body, _name, parent_name, frame_offset, rt_in_matrix = 0,
                             true_segment=False, _dof_total_trans='', _dof_total_rot='', _range_q = ''):
            if rt_in_matrix not in [0, 1]:
                raise ('Error: rt_in_matrix can be only set to 1 or 0')
            if rt_in_matrix == 1:
                print('not implemented yet')
                [[r11, r12, r13, r14],
                 [r21, r22, r23, r24],
                 [r31, r32, r33, r34],
                 [r41, r42, r43, r44]] = frame_offset.get_matrix().tolist()

                for i in range(4):
                    for j in range(4):
                        round(eval('r' + str(i + 1) + str(j + 1)), 9)

            range_q_text = ''
            for i in range_q:
                range_q_text += f'               {i[0]}\t{i[1]}\n'
            [i11, i22, i33, i12, i13, i23] = matrix_inertia(_body)
            mass = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'mass'))
            com = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'mass_center'))
            path_mesh_file = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'mesh_file'))
            # TODO add mesh files

            # writing data
            self.write('    // Segment\n')
            self.write('    segment {}\n'.format(_name)) if _name != 'None' else self.write('')
            self.write('        parent {} \n'.format(parent_name)) if (parent_name != 'None' or parent_name != 'ground') else self.write('')
            self.write('        RTinMatrix    {}\n'.format(rt_in_matrix)) if rt_in_matrix != 'None' else self.write('')
            if rt_in_matrix == 0:
                self.write('        RT\t{}\txyz\t{}\n'.
                           format(' '.join(map(str, frame_offset[1])), ' '.join(map(str, frame_offset[0]))))
            else:
                self.write('        RT\n')
                self.write(
                    '            {}    {}    {}    {}\n'
                    '            {}    {}    {}    {}\n'
                    '            {}    {}    {}    {}\n'
                    '            {}    {}    {}    {}\n'
                        .format(r11, r12, r13, r14,
                                r21, r22, r23, r24,
                                r31, r32, r33, r34,
                                r41, r42, r43, r44))
            self.write('        translations\t{}\n'.format(
                _dof_total_trans)) if _dof_total_trans != '' else True
            self.write('        rotations\t{}\n'.format(_dof_total_rot)) if _dof_total_rot != '' else True
            if _range_q != '':
                self.write('        rangesQ\n'
                           '{}'.format(range_q_text)) if _range_q != '' else True
            self.write('        mass {}\n'.format(mass)) if true_segment is True else True
            self.write('        inertia\n'
                       '            {}    {}    {}\n'
                       '            {}    {}    {}\n'
                       '            {}    {}    {}\n'
                       .format(i11, i12, i13,
                               i12, i22, i23,
                               i13, i23, i33)) if true_segment is True else True
            self.write('        com    {}\n'.format(com)) if true_segment is True else True
            self.write('        meshfile Meshfiles/{}\n'
                       .format(path_mesh_file)) if path_mesh_file != 'None' and true_segment else True
            self.write('    endsegment\n')

        # Division of body in segment depending of transformation and child_offset
        # if the child has an offset, two segments are created, the first represents the dofs,
        # while the second carries inertia information, and the child offset
        dof_chain = '\n//BREAKDOWN OF KINEMATIC CHAIN'
        for body in body_list(self):
            rotomatrix = OrthoMatrix([0, 0, 0])
            self.write('\n// Information about {} segment\n'.format(body))
            parent = parent_child(body)
            joint, joint_type = joint_body(body)
            list_transform = transform_of_joint(joint, joint_type)
            frame_offset = get_frames_offsets(joint, joint_type)

            #Todo:
            # -improve the transform function choice using goto function
            # -combine translation and rotation in one
            # -include Pinjoint in the constrained coordinates?

            # get dof



            if (joint_type == "PinJoint"):
                dof_total_trans = ''
                dof_total_rot = 'z'
            else:
                range_q = []
                # translation dof

                if list_transform[0] == []:
                    dof_total_trans = ''
                else:
                    additional_dof_output, range_q_translation, dof_total_trans = extract_information_dof(list_transform[0],
                                                                                         'translation',
                                                                                         joint_type, joint, range_q)
                    dof_chain += additional_dof_output

                if list_transform[1] == []:
                    dof_total_rot = ''
                else:
                    additional_dof_output, range_q_translation, dof_total_rot = extract_information_dof(list_transform[1],
                                                                                         'rotation',
                                                                                         joint_type, joint, range_q)
                    dof_chain += additional_dof_output

            if frame_offset[1]:
                self.write('\n\t// Two segments are used for {} in Opensim to express the joint''s child_offset\n'.format(body))
                printing_segment(body, body+'_offset', parent, frame_offset[0], rt_in_matrix=0, true_segment=False,
                                 _dof_total_trans=dof_total_trans, _dof_total_rot=dof_total_rot, _range_q=range_q)

                printing_segment(body, body, body+'_offset', frame_offset[1], rt_in_matrix=0, true_segment=True)

            else:
                printing_segment(body, body, parent, frame_offset[0], rt_in_matrix=0,true_segment=True,
                                 _dof_total_trans=dof_total_trans, _dof_total_rot=dof_total_rot, _range_q = range_q)

            ###########    Markers extraction

            _list_markers = list_markers_body(body)
            if _list_markers is not []:
                self.write('\n    // Markers')
                for marker in _list_markers:
                    position = new_text(go_to(go_to(self.root, 'Marker', 'name', marker), 'location'))
                    self.write('\n    marker    {}'.format(marker))
                    self.write('\n        parent    {}'.format(parent))
                    self.write('\n        position    {}'.format(position))
                    self.write('\n    endmarker\n')
            late_body = body

        ###########    Coupled Coordinates constraints

        constraints_output = '\n//COUPLED COORDINATES\n\n'
        cc_constraints = index_go_to(self.root, 'CoordinateCouplerConstraint')
        if cc_constraints is not None:

            constraints_in_model = True
            list_index = list(cc_constraints)
            cc_index = ''.join(list_index[:len(list_index) - 2])
            i = 0
            while True:
                try:
                    new_ccc = eval('self.root' + cc_index + str(i) + ']').get('name')
                    constraints_output += f'\n//name: {new_ccc}\n'
                    qx = new_text(go_to(go_to(self.root, 'CoordinateCouplerConstraint', 'name', new_ccc),
                                        'independent_coordinate_names'))
                    qy = new_text(go_to(go_to(self.root, 'CoordinateCouplerConstraint', 'name', new_ccc),
                                        'dependent_coordinate_name'))
                    type_coupling = go_to(go_to(self.root, 'CoordinateCouplerConstraint', 'name', new_ccc),
                                          'coupled_coordinates_function')[0].tag
                    constraints_output += f'\t//independent q: {qx}\n' \
                        f'\t//dependent q: {qy}\n' \
                        f'\t//coupling type: {type_coupling}\n'
                    i += 1
                except:
                    break

        ###########    Muscle Definition

        self.write('\n// MUSCLE DEFINIION\n')
        sort_muscle = []
        muscle_ref_group = []
        for muscle in muscle_list(self):
            viapoint = list_pathpoint_muscle(muscle)
            bodies_viapoint = []
            for pathpoint in viapoint:
                bodies_viapoint.append(get_body_pathpoint(pathpoint))
            # Todo: in Osim STFD upper limb model this isn't true, is it worth changing?
            # it is supposed that viapoints are organized in order
            # from the parent body to the child body
            body_start = bodies_viapoint[0]
            body_end = bodies_viapoint[len(bodies_viapoint) - 1]
            sort_muscle.append([body_start, body_end])
            muscle_ref_group.append([muscle, body_start + '_to_' + body_end])
        # selecting muscle group
        group_muscle = []
        for ext_muscle in sort_muscle:
            if ext_muscle not in group_muscle:
                group_muscle.append(ext_muscle)
                # print muscle group
        for muscle_group in group_muscle:
            self.write('\n// {} > {}\n'.format(muscle_group[0], muscle_group[1]))
            self.write('musclegroup {}\n'.format(muscle_group[0] + '_to_' + muscle_group[1]))
            self.write('    OriginParent        {}\n'.format(muscle_group[0]))
            self.write('    InsertionParent        {}\n'.format(muscle_group[1]))
            self.write('endmusclegroup\n')
            # muscle
            for muscle in muscle_list(self):
                # muscle data
                m_ref = muscle_group_reference(muscle, muscle_ref_group)
                if m_ref == muscle_group[0] + '_to_' + muscle_group[1]:
                    muscle_type = 'hillthelen'
                    state_type = 'buchanan'
                    list_pathpoint = list_pathpoint_muscle(muscle)
                    start_point = list_pathpoint.pop(0)
                    end_point = list_pathpoint.pop()
                    start_pos = get_pos(start_point)
                    insert_pos = get_pos(end_point)
                    opt_length = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'optimal_fiber_length'))
                    max_force = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'max_isometric_force'))
                    tendon_slack_length = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'tendon_slack_length'))
                    pennation_angle = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'pennation_angle_at_optimal'))
                    pcsa = new_text(go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'pcsa'))
                    max_velocity = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'max_contraction_velocity'))

                    # print muscle data
                    self.write('\n    muscle    {}'.format(muscle))
                    self.write('\n        Type    {}'.format(muscle_type)) if muscle_type != 'None' else self.write('')
                    self.write('\n        statetype    {}'.format(state_type)) if state_type != 'None' else self.write(
                        '')
                    self.write('\n        musclegroup    {}'.format(m_ref)) if m_ref != 'None' else self.write('')
                    self.write(
                        '\n        OriginPosition    {}'.format(start_pos)) if start_pos != 'None' else self.write('')
                    self.write(
                        '\n        InsertionPosition    {}'.format(insert_pos)) if insert_pos != 'None' else self.write(
                        '')
                    self.write(
                        '\n        optimalLength    {}'.format(opt_length)) if opt_length != 'None' else self.write('')
                    self.write('\n        maximalForce    {}'.format(max_force)) if max_force != 'None' else self.write(
                        '')
                    self.write('\n        tendonSlackLength    {}'.format(
                        tendon_slack_length)) if tendon_slack_length != 'None' else self.write('')
                    self.write('\n        pennationAngle    {}'.format(
                        pennation_angle)) if pennation_angle != 'None' else self.write('')
                    self.write('\n        PCSA    {}'.format(pcsa)) if pcsa != 'None' else self.write('')
                    self.write(
                        '\n        maxVelocity    {}'.format(max_velocity)) if max_velocity != 'None' else self.write(
                        '')
                    self.write('\n    endmuscle\n')
                    # viapoint
                    for viapoint in list_pathpoint:
                        # viapoint data
                        parent_viapoint = get_body_pathpoint(viapoint)
                        viapoint_pos = get_pos(viapoint)
                        # print viapoint data
                        self.write('\n        viapoint    {}'.format(viapoint))
                        self.write('\n            parent    {}'.format(
                            parent_viapoint)) if parent_viapoint != 'None' else self.write('')
                        self.write('\n            muscle    {}'.format(muscle))
                        self.write('\n            musclegroup    {}'.format(m_ref)) if m_ref != 'None' else self.write(
                            '')
                        self.write('\n            position    {}'.format(
                            viapoint_pos)) if viapoint_pos != 'None' else self.write('')
                        self.write('\n        endviapoint')
                    self.write('\n')


        ###########    Additional data as comment about the model's constraints
        if constraints_in_model:
            self.write(constraints_output)
            self.write(dof_chain)
            self.write_insert(3, '\n\nWARNING\n'
                                 '// The original model has some constrained DOF, thus it can not be '
                                 'directly used for kinematics or dynamics analysis.\n'
                                 '//If used in optimization, constraints should be added to the nlp to account'
                                 ' for the reduced number of DOF\n'
                                 '// Check end of file for possible constraints in the osim model\n\n')

        self.file.close()

    def __getattr__(self, attr):
        print('Error : {} is not an attribute of this class'.format(attr))

    def get_path(self):
        return self.path

    def write(self, string):
        self.file = open(self.path, 'a')
        self.file.write(string)
        self.file.close()

    def write_insert(self, line_index, string):
        self.file = open(self.path, "r")
        contents = self.file.readlines()
        self.file.close()

        contents.insert(line_index, string)

        self.file = open(self.path, 'w')
        contents = "".join(contents)
        self.file.write(contents)
        self.file.close()

    def get_origin_file(self):
        return self.originfile

    def credits(self):
        return self.data_origin.xpath(
            '/OpenSimDocument/Model/credits')[0].text

    def publications(self):
        return self.data_origin.xpath(
            '/OpenSimDocument/Model/publications')[0].text

    def body_list(self):
        _list = []
        for body in self.data_origin.xpath(
                '/OpenSimDocument/Model/BodySet/objects/Body'):
            _list.append(body.get("name"))
        return _list
